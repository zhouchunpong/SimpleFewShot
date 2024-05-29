import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from collections import OrderedDict
import argparse
import numpy as np
import random
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
 
from torch.cuda.amp import autocast as autocast


import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

#from sentence_transformers import SentenceTransformer
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import visformer
from data.dataloader import EpisodeSampler, MultiTrans
from data.dataset import DatasetWithTextLabel
from data.randaugment import RandAugmentMC
from utils import mean_confidence_interval

  
import time
print (time.strftime("%S",time.localtime()))

 

def main(args):
    # checkpoint and tensorboard dir
    args.tensorboard_dir = 'tensorboard/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    args.checkpoint_dir = 'checkpoint/' + args.dataset + '/' + args.model + '/' + args.exp + '/'
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.logger = SummaryWriter(args.tensorboard_dir)

    # prepare training and testing dataloader
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    train_aug = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    norm])
    if args.aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        norm])
    if args.rand_aug:
        train_aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
                                        RandAugmentMC(2, 10, args.image_size),
                                        transforms.ToTensor(),
                                        norm])
    test_aug = transforms.Compose([transforms.Resize(int(args.image_size * 1.1)),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   norm])
    # if args.aug_support > 1:
    #     aug = transforms.Compose([transforms.RandomResizedCrop(args.image_size),
    #                               # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #                               transforms.RandomHorizontalFlip(),
    #                               transforms.ToTensor(),
    #                               norm])
    #     test_aug = MultiTrans([test_aug] + [aug]*(args.aug_support-1))

    train_dataset = DatasetWithTextLabel(args.dataset, train_aug, split='train')
    n_episodes = args.train_episodes
    args.train_way = args.way if args.train_way == -1 else args.train_way
    if n_episodes == -1:
        n_episodes = int(len(train_dataset) / (args.train_way * (args.shot + 15)))
      
    episode_sampler = EpisodeSampler(train_dataset.dataset.targets,
                                     n_episodes,
                                     args.train_way,
                                     args.shot + 15, fix_seed=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=episode_sampler, num_workers=8)
    num_classes = len(train_dataset.dataset.classes)

    test_dataset = DatasetWithTextLabel(args.dataset, test_aug, split='test')
    episode_sampler = EpisodeSampler(test_dataset.dataset.targets, args.test_episodes, args.way, args.shot + 15)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=episode_sampler, num_workers=6)

    #gpu_tracker.track()
    # load CLIP model
    teacher, _ = clip.load("ViT-B/32", device='cuda:' + str(args.gpu))
    teacher.float()
    teacher.requires_grad_(False)
    teacher.eval()
    text_dim = 512

    
    train_class_idx = train_dataset.dataset.classes
    train_idx2text = train_dataset.idx2text
    train_classnames = [train_idx2text[idx] for idx in train_class_idx]

    test_class_idx = test_dataset.dataset.classes
    test_idx2text  = test_dataset.idx2text
    test_classnames = [test_idx2text[idx] for idx in test_class_idx]

    all_classnames =  train_classnames + test_classnames
 
    text_encoder = TextEncoder(teacher).cuda()
    prompt_learner = PromptLearner(all_classnames, teacher).cuda()
    tokenized_prompts = prompt_learner.tokenized_prompts  #fixed
    prompts = prompt_learner()
    
    
    student = visformer.visformer_tiny(num_classes=num_classes, drop_rate = args.dropout)


    feature_dim = 384
    if 2 <= args.stage < 3:
        feature_dim = 192
    


    if args.adaptor == 'linear':
        student.adaptor = nn.Linear(text_dim, feature_dim, bias=False)
    elif args.adaptor == 'mlp':
        student.adaptor = nn.Sequential(
                        nn.Linear(text_dim,feature_dim//4),
                        nn.LeakyReLU(),
                        nn.Linear(feature_dim//4,feature_dim)
                        )
    else:
        student.adaptor = adaptor(text_dim,feature_dim)


    student = student.cuda(args.gpu)

    optim_params_id = [id(param) for param in student.adaptor.parameters()]


    optim_params = [param for param in student.parameters() if id(param) in optim_params_id] 
    other_params = [param for param in student.parameters() if id(param) not in optim_params_id]

    optim = torch.optim.AdamW([{'params': optim_params, 'lr': args.lr},
                                   {'params': other_params, 'lr': args.encoder_lr},
                                   {'params': prompt_learner.parameters(), 'lr': args.lr}], weight_decay=args.weight_decay)


    if args.resume:
        args.init = args.resume
    if args.init:
        checkpoint = torch.load(args.init, map_location=f'cuda:{args.gpu}')
        student.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        raise ValueError('must provide pre-trained model')

    start_epoch = 0


    if args.test:
        test(prompt_learner, prompts,tokenized_prompts, text_encoder,student, test_loader, epoch, args)
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode = 'max',factor = 0.1,patience = 50,verbose=True)

    best_acc = 0.
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        
        train(prompt_learner,prompts,tokenized_prompts,text_encoder,student, train_loader, optim, epoch,args)


        if (epoch + 1) % args.test_freq == 0:
            acc = test(prompt_learner,prompts,tokenized_prompts,text_encoder,student, test_loader, epoch, args)

        if args.sheduler == 'True':
            scheduler.step(acc)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': student.state_dict(),
            'optimizer': optim.state_dict(),
            'prompt_learner':prompt_learner.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_latest.pth')
        if (epoch + 1) % args.save_freq == 0:
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_{epoch + 1:03d}.pth')
        if (epoch + 1) % args.test_freq == 0 and acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(checkpoint, args.checkpoint_dir + f'checkpoint_epoch_best.pth')
        
        print("best_epoch: %d, best_acc: %.4f" %(best_epoch,best_acc) )



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
        #ic(x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD 
        x = self.ln_final(x)
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class adaptor(nn.Module):
    def __init__(self,text_dim, feature_dim):
        super().__init__()
        self.layer1 = nn.Linear(text_dim, feature_dim)
        self.layer2 = nn.Linear(feature_dim,feature_dim//4)
        self.layer3 = nn.Linear(feature_dim//4,feature_dim)

    def forward(self,x):
        x = F.leaky_relu(self.layer1(x))
        x1 = F.leaky_relu(self.layer2(x))
        x2 = self.layer3(x1) + x
        return x2





class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = 16   # number of context vectors
        ctx_init = "a photo of a"  
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
       
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        #token_embedding_ = clip_model.token_embedding.cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        return prompts


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)/32
    return logits


def JS_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss()
    
    p_output = F.softmax(p_output)
    q_output = F.softmax(q_output)

    log_mean_output = ((p_output + q_output )/2).log()
    
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2



def train(prompt_learner, prompts,tokenized_prompts, text_encoder,student, train_loader, optim, epoch,args):
    student.train()
    prompt_learner.train()
    
    losses = 0.
    acc_im = 0.
    acc_text = 0.
    # acc_cat = 0.
    # acc_sum = 0.
    acc_sum2 = 0.
    acc_sum4 = 0.
    acc_sum6 = 0.
    acc_sum8 = 0.
    acc_sum10 = 0.

    for idx, episode in enumerate(train_loader):
        image = episode[0].cuda(args.gpu)  # way * (shot+15)
        glabels = episode[1].cuda(args.gpu)
        labels = torch.arange(args.train_way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)


        image = image.view(args.train_way, args.shot+15, *image.shape[1:])
        sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
        sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])
  

        glabels = glabels.view(args.train_way, args.shot+15)[:, :args.shot]
        glabels = glabels.contiguous().view(-1)


        text_features = text_encoder(prompts[glabels],tokenized_prompts[glabels])
        avg_length = (text_features ** 2).sum(-1).sqrt().mean().item()
        text_features = F.normalize(text_features, dim=-1) * avg_length
        
        _,sup_im_features = student(sup)
        _, que_im_features = student(que)
        que_im_features = F.normalize(que_im_features, dim=-1)


        im_proto = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)
        im_proto = F.normalize(im_proto, dim=-1)
        sim_im = que_im_features @ im_proto.t()


        text_features = student.adaptor(text_features)
        sup_im_features = sup_im_features + text_features
        sup_im_features = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)

        sup_im_features = F.normalize(sup_im_features, dim=-1) # [5, 384]
        
        sim_text =  torch.mm(que_im_features,sup_im_features.t())

        sim_sum2 = sim_text + 0.2*sim_im

        sim_sum4 = sim_text + 0.4*sim_im

        sim_sum6 = sim_text + 0.6*sim_im

        sim_sum8 = sim_text + 0.8*sim_im

        sim_sum10 = sim_text + sim_im

        
        KD_loss = JS_div(sim_im/ args.t,sim_text/ args.t) 
    
        loss = F.cross_entropy(sim_im / args.t, labels) + \
            F.cross_entropy(sim_text / args.t, labels)  + \
            args.KD*KD_loss
        
        losses += loss.item()

        _, pred = sim_im.max(-1)
        acc_im += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_text.max(-1)
        acc_text += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_sum2.max(-1)
        acc_sum2 += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_sum4.max(-1)
        acc_sum4 += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_sum6.max(-1)
        acc_sum6 += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_sum8.max(-1)
        acc_sum8 += labels.eq(pred).sum().float().item() / labels.shape[0]

        _, pred = sim_sum10.max(-1)
        acc_sum10 += labels.eq(pred).sum().float().item() / labels.shape[0]

        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()


        if idx % args.print_step == 0 or idx == len(train_loader) - 1:
            print_string = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, acc_im: {acc_im * 100 / (idx + 1):.2f},, acc_text: {acc_text * 100 / (idx + 1):.2f},, acc_sum2: {acc_sum2 * 100 / (idx + 1):.2f},acc_sum4: {acc_sum4 * 100 / (idx + 1):.2f},acc_sum6: {acc_sum6 * 100 / (idx + 1):.2f},acc_sum8: {acc_sum8 * 100 / (idx + 1):.2f},acc_sum10: {acc_sum10 * 100 / (idx + 1):.2f}'
            print(print_string)
            
    args.logger.add_scalar('train/loss', losses / len(train_loader), epoch)
    args.logger.add_scalar('train/acc', acc_sum2 / len(train_loader), epoch)


def test(prompt_learner, prompts,tokenized_prompts, text_encoder,student, test_loader, epoch, args):
    student.eval()
    prompt_learner.eval()
    acc_im = []
    acc_text = []
    acc_cat = []
    acc_sum2 = []
    acc_sum4 = []
    acc_sum6 = []
    acc_sum8 = []
    acc_sum10 = []

    with torch.no_grad():
        for episode in test_loader:
            # use prototype classifier
            image = episode[0].cuda(args.gpu)  # way * (shot+15)
            glabels = episode[1].cuda(args.gpu)
            labels = torch.arange(args.way).unsqueeze(-1).repeat(1, 15).view(-1).cuda(args.gpu)

            image = image.view(args.way, args.shot + 15, *image.shape[1:])
            sup, que = image[:, :args.shot].contiguous(), image[:, args.shot:].contiguous()
            sup, que = sup.view(-1, *sup.shape[2:]), que.view(-1, *que.shape[2:])

            glabels = glabels.view(args.train_way, args.shot+15)[:, :args.shot]
            glabels = glabels.contiguous().view(-1)

            
            text_features = text_encoder(prompts[glabels+args.delta],tokenized_prompts[glabels+args.delta])
            avg_length = (text_features ** 2).sum(-1).sqrt().mean().item()
            text_features = F.normalize(text_features, dim=-1) * avg_length
            

            _,sup_im_features = student(sup)
            _, que_im_features = student(que)

            que_im_features = F.normalize(que_im_features, dim=-1) # [75, 384]

            im_proto = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)
            im_proto = F.normalize(im_proto, dim=-1)
            sim_im = que_im_features @ im_proto.t()

            text_features = student.adaptor(text_features)
            sup_im_features = sup_im_features + text_features
            sup_im_features = sup_im_features.view(args.train_way, args.shot, -1).mean(dim=1)

            sup_im_features = F.normalize(sup_im_features, dim=-1) # [5, 384]
            
            sim_text =  torch.mm(que_im_features,sup_im_features.t())

 
            sim_sum2 = sim_text + 0.2*sim_im

            sim_sum4 = sim_text + 0.4*sim_im

            sim_sum6 = sim_text + 0.6*sim_im

            sim_sum8 = sim_text + 0.8*sim_im

            sim_sum10 = sim_text + sim_im

            _, pred = sim_im.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_im.append(acc)

            _, pred = sim_text.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_text.append(acc)

            _, pred = sim_sum2.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_sum2.append(acc)

            _, pred = sim_sum4.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_sum4.append(acc)

            _, pred = sim_sum6.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_sum6.append(acc)

            _, pred = sim_sum8.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_sum8.append(acc)

            _, pred = sim_sum10.max(-1)
            acc = labels.eq(pred).sum().float().item() / labels.shape[0]
            acc_sum10.append(acc)

            

    m, h = mean_confidence_interval(acc_im)
    print(f'sim Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    m, h = mean_confidence_interval(acc_text)
    print(f'text Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    
    m, h = mean_confidence_interval(acc_sum2)
    print(f'acc_sum2 Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    m, h = mean_confidence_interval(acc_sum4)
    print(f'acc_sum4 Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    m, h = mean_confidence_interval(acc_sum6)
    print(f'acc_sum6 Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    m, h = mean_confidence_interval(acc_sum8)
    print(f'acc_sum8 Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')
    m, h = mean_confidence_interval(acc_sum10)
    print(f'acc_sum10 Test epoch: {epoch}, test acc: {m * 100:.2f}+-{h * 100:.2f}')

    args.logger.add_scalar('test/acc', m * 100, epoch)

    return m



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR-FS', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--image_size', type=int, default=224, choices=[224, 84])
    parser.add_argument('--aug', action='store_true', default=True)
    parser.add_argument('--rand_aug', action='store_true')
    parser.add_argument('--aug_support', type=int, default=1)
    parser.add_argument('--model', type=str, default='visformer-t', choices=['visformer-t', 'visformer-t-84','res'])
    parser.add_argument('--nlp_model', type=str, default='clip', choices=['clip', 'glove', 'mpnet'])
    parser.add_argument('--prompt_mode', type=str, default='add', choices=['add','spatial', 'channel', 'spatial+channel'])
    parser.add_argument('--no_template', action='store_true')
    parser.add_argument('--eqnorm', action='store_true', default=True)
    parser.add_argument('--stage', type=float, default=3.2, choices=[2, 2.1, 2.2, 2.3, 3, 3.1, 3.2, 3.3])
    parser.add_argument('--projector', type=str, default='linear', choices=['linear', 'mlp', 'mlp3'])
    parser.add_argument('--avg', type=str, default='all', choices=['all', 'patch', 'head'])
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adamw', choices=['sgd', 'adamw'])
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--encoder_lr', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='checkpoint/cifar/checkpoint_epoch_800.pth')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--text_length', type=int, default=20)
    parser.add_argument('--train_way', type=int, default=-1)  
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=-1)
    parser.add_argument('--test_episodes', type=int, default=2000)
    parser.add_argument('--test_classifier', type=str, default='prototype', choices=['prototype', 'fc'])
    parser.add_argument('--print_step', type=int, default=300)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--comment', type=str, default=' ')
    parser.add_argument('--sheduler', type=str, default='False')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--KD', type=float, default=1)
    parser.add_argument('--adaptor', type=str, default='mlp', choices=['linear', 'mlp', 'bottle'])
    

    args = parser.parse_args()
    from datetime import datetime

    now = datetime.now()
   

    args.exp =  args.exp + args.dataset + str(now)[:18] + args.comment

    # ['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    if args.dataset == 'FC100':
        args.delta = 60
    elif args.dataset == 'tieredImageNet':
        args.delta = 351
    else:
        args.delta = 64


    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    main(args)


