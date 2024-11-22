import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import matplotlib.pyplot as plt

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class NegaPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_nega_ctx = cfg.TRAINER.COOP.NEGA_CTX
        self.n_nega_ctx = n_nega_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            ctx_vectors = ctx_vectors.view(1, ctx_vectors.shape[0], ctx_vectors.shape[1])
            ctx_vectors = ctx_vectors.repeat(1 + n_nega_ctx, 1, 1)
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(1 + n_nega_ctx, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if ctx_vectors.dim() == 3:
            ctx_positive = ctx_vectors[0:1, :, :]
            ctx_negative = ctx_vectors[1:, :, :]
        else:
            ctx_positive = ctx_vectors[:, 0:1, :, :]
            ctx_negative = ctx_vectors[:, 1:, :, :]
        self.ctx_positive = nn.Parameter(ctx_positive)
        self.ctx_negative = nn.Parameter(ctx_negative)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " + name for name in classnames]
        negative_prompts = [prompt_prefix + " " + name for name in classnames]

        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts])
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype)

        positive_embedding = positive_embedding.view(positive_embedding.shape[0], 1, positive_embedding.shape[1], positive_embedding.shape[2])
        negative_embedding = negative_embedding.view(negative_embedding.shape[0], 1, negative_embedding.shape[1], negative_embedding.shape[2])
        negative_embedding = negative_embedding.repeat(1, n_nega_ctx, 1, 1)
        embedding = torch.cat([positive_embedding, negative_embedding], dim=1)
        positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0], 1, positive_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0], 1, negative_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.repeat(1, n_nega_ctx, 1)
        tokenized_prompts = torch.cat([positive_tokenized_prompts, negative_tokenized_prompts], dim=1)
        tokenized_prompts = tokenized_prompts.view(tokenized_prompts.shape[0] * tokenized_prompts.shape[1], -1)

        self.register_buffer("token_prefix", embedding[:, :, :1, :])
        self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, modify_to_ori=None):
        ctx_positive = self.ctx_positive
        ctx_negative = self.ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2])).to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if modify_to_ori is not None:
            ori_labels = list(modify_to_ori.values())
            ctx = ctx[ori_labels]
            prefix = prefix[ori_labels]
            suffix = suffix[ori_labels]
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=2,
        )

        return prompts
    
    def forward_negative(self):
        ctx_negative = self.ctx_negative
        if ctx_negative.dim() == 3:
            ctx = ctx_negative.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_negative
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=2,
        )
        return prompts

class NegaTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.eval()
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        if(hasattr(clip_model, 'attn_mask')):
            self.attn_mask = clip_model.attn_mask
        else:
            self.attn_mask = None
        # print('attn_mask is ', self.attn_mask)
    
    def forward(self, prompts, tokenized_prompts):
        '''
        Encodes the given text prompts using the CLIP transformer.
        '''
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (n_class*(1+n_neg)) * n_ctx * dim 
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)
        # x = self.transformer(x, self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print("x shape: ", x.shape)
        # print("tokenized_prompts shape: ", tokenized_prompts.shape)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class NegaPromptCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = NegaPromptLearner(cfg, classnames, clip_model).cuda()
        self.n_nega_ctx = cfg['NEGA_CTX']
        self.stage = cfg['stage']
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = NegaTextEncoder(clip_model).cuda()
        # self.text_encoder = TextEncoder(clip_model)
        # self.weight_yes = self.merge_yes_feature(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.positive_text_features = None
        self.clip_model = clip_model
        self.cfg = cfg

    def forward_negative(self, image):
        '''
        Only learn the negative prompts
        return shape:
        logits: [batch_size, nclass * 1+n_nega_ctx]
        text_features: [nclass * 1+n_nega_ctx, 512]
        '''
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        negative_prompts = self.prompt_learner.foward_negative()    # use negative prompts only
        negative_tokenized_prompts = self.prompt_learner.negative_tokenized_prompts
        negative_text_features = self.text_encoder(negative_prompts, negative_tokenized_prompts) #(1000*n_nega_ctx) * 512)
        positive_text_features = self.positive_text_features # 1000*512, fixed
        #fusion the text_features that positive, negative, positive, negative, ...
        positive_text_features = positive_text_features.view(positive_text_features.shape[0], 1, -1)
        negative_text_features = negative_text_features.view(positive_text_features.shape[0], self.n_nega_ctx, -1)  # 1000 * n_nega_ctx * 512

        # here we concatenate the positive and negative text features
        text_features = torch.cat([positive_text_features, negative_text_features], dim=1)  
        text_features = text_features.view(text_features.shape[0]*text_features.shape[1], -1)   # 1000*(1+n_nega_ctx) * 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # shape: 1000*(1+n_nega_ctx) * 512
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features
    
    def forward(self, image, modify_to_ori = None):
        '''
        If stage == 3, only learn the negative prompts, otherwise learn both positive and negative prompts
        Return the logits and text embeddings by CLIP
        '''
        if self.stage == 3:
            return self.forward_negative(image) # only learn the negative prompts
        
        # otherwise, learn both positive and negative prompts
        prompts = self.prompt_learner(modify_to_ori)
        # prompt shape: [n_class, 1+n_neg, n_ctx, dim]
        tokenized_prompts = self.tokenized_prompts
        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features
    
    def forward_test(self, image, text_features=None):
        '''The forward method for testing, need input trianed text_features'''
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features

    def get_visual_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

            
    def get_ctx_posi(self, ctx_posi):
        '''Use ctx_posi to update the positive context vectors, and generate negative context vectors.
        Then get the positive text features by CLIP text encoder into positive_text_features.
        '''
        self.prompt_learner.update_ctx_positive(ctx_posi)
        # get positive_text_features
        prompts = self.prompt_learner.foward_positive() # Returns the prompt vectors for the positive class names.
        tokenized_prompts = self.prompt_learner.positive_tokenized_prompts
        self.positive_text_features = self.text_encoder(prompts, tokenized_prompts) # get text embedding for positive prompts by CLIP transformer

    def get_ctx_nega(self, ctx_nega):
        '''Set the negative context vectors to ctx_nega.'''
        self.prompt_learner.update_ctx_negative(ctx_nega)
    
    def freeze_ctx_posi(self):   # not used. There are other functions doing this
        '''
        Freeze the positive context vectors to self.ctx_positive.
        '''
        self.prompt_learner.freeze_ctx_positive()

    def radius(self):# not used, maybe in loss calculation
        ''' calculate the cos distance between positive and negative text features, and return the radius'''
        prompts = self.prompt_learner() 
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        n_nega_ctx = self.cfg['NEGA_CTX']
        ensemble_text_features = text_features.view(int(text_features.shape[0]/(1+n_nega_ctx)), 1+n_nega_ctx, -1)
        positive_text_features = ensemble_text_features[:, 0, :]
        negative_text_features = ensemble_text_features[:, 1:, :]
        radius = torch.Tensor(positive_text_features.shape[0], n_nega_ctx)
        logit_scale = self.logit_scale.exp()
        for i in range(positive_text_features.shape[0]):
            positive_feature = positive_text_features[i,:]
            negative_features = negative_text_features[i,:,:]
            
            cos_sim = torch.nn.functional.cosine_similarity(negative_features, positive_feature.unsqueeze(0), dim=1)
            one_radius = 1 - cos_sim
            
            # one_radius = logit_scale*positive_feature @ negative_features.t()
            
            radius[i, :] = one_radius
        
        
        return radius
    
    def draw_tsne_plot(self, testloader, outloader, log_dir, expr_name, epoch):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.reshape(prompts.shape[0], prompts.shape[1], text_features.shape[-1])
        pos_feature = text_features[:, 0:1, :].cpu()
        pos_feature = pos_feature / pos_feature.norm(dim=-1, keepdim=True)
        neg_feature = text_features[:, 1:, :].cpu()
        neg_feature = neg_feature / neg_feature.norm(dim=-1, keepdim=True)
        pos_label = torch.arange(pos_feature.shape[0])[..., None] # shape = [nclass, 1]
        neg_label = torch.full((neg_feature.shape[0], neg_feature.shape[1]), pos_feature.shape[0]) #shape = [nclass, n_nega]

        n_class = pos_feature.shape[0]
        
        all_image_feature = torch.Tensor()
        all_image_label = torch.Tensor()
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                image_features = self.image_encoder(data.type(self.dtype)).cpu()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
                all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)
                

        all_text_feature = torch.Tensor()               
        all_text_feature = torch.cat([all_text_feature, pos_feature], dim=1)
        all_text_feature = all_text_feature.view(-1, all_text_feature.shape[-1])
        
        all_text_label = torch.Tensor()
        all_text_label = torch.cat([all_text_label, pos_label], dim=1)
        all_text_label = all_text_label.view(-1)
        
        total_feature = torch.cat([all_text_feature, all_image_feature], dim=0)
        total_label = torch.cat([all_text_label, -1 * (all_image_label+1)], dim=0)

        X = total_feature.detach().numpy()
        tsne_model = TSNE(metric="precomputed", n_components=2, init="random", perplexity=30)
        distance_matrix = pairwise_distances(X, X, metric='cosine', n_jobs=-1)
        
        data = torch.Tensor(tsne_model.fit_transform(distance_matrix))
        target = total_label
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=256)
        plt.figure()
        for x, y in loader:
            # 样本点显示
            idx_pos_text = (y < n_class) & (y >= 0)  # 正向样本 
            idx_nega_text = (y >= n_class)  # 负向样本
            idx_pos_image = (y < 0) & (y >= -n_class)
            idx_nega_image = (y < -n_class)

            plt.scatter(x[idx_pos_text, 0], x[idx_pos_text, 1], marker = 'o', c=y[idx_pos_text], alpha=0.2,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='pos')
            plt.scatter(x[idx_nega_text, 0], x[idx_nega_text, 1], marker = 'o', c=y[idx_nega_text], alpha=0.2,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='nega')
            plt.scatter(x[idx_pos_image, 0], x[idx_pos_image, 1], marker = 'x',c =-1 * y[idx_pos_image] - 1, alpha=0.4,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='pos')
            plt.scatter(x[idx_nega_image, 0], x[idx_nega_image, 1], marker = 'x',c=-1 * y[idx_nega_image] - 1, alpha=0,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='nega')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles, labels)
        dir_path = os.path.join(log_dir, 'tsne', expr_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        plt.savefig(os.path.join(dir_path, 'tsne_plot_epoch_{}.pdf'.format(epoch)))
        plt.close()


@TRAINER_REGISTRY.register()
class NegPromptTrainer(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = NegaPromptCLIP(cfg.TRAINER.COOP, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler('cuda') if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)