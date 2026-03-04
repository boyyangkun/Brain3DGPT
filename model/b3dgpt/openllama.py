import torch
import numpy as np
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from .Brain3DGPT_models import  MultiScaleAnomalyGuidedFusion
from ..Uni_unet import UniUnet
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn.utils import rnn
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):
    def __init__(self, args):
        super(OpenLLAMAPEFTModel, self).__init__()
        llama_ckpt_path = args.llama_ckpt_path
        self.input_shape = args.patch_shape
        self.in_channels = args.in_channels
        self.out_channels = args.out_classes
        self.max_tgt_len = args.max_tgt_len
        print (f'Initializing image decoder...')
        self.UniUnet_model = UniUnet(input_shape = self.input_shape, in_channels = self.in_channels, out_channels = self.out_channels, multi_scale = True)
        ckpt = torch.load(args.pretrained, map_location=torch.device('cpu'))
        self.UniUnet_model.load_state_dict(ckpt['state_dict'], strict=False)
        self.UniUnet_model.eval()

        print (f'Initializing language decoder from {llama_ckpt_path} ...')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_ckpt_path) 
        self.llama_model = get_peft_model(self.llama_model, peft_config)

        self.llama_model.print_trainable_parameters()
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

        self.MultiScaleFusion = MultiScaleAnomalyGuidedFusion(64, 128, 320, 512, self.llama_model.config.hidden_size)

        # 冻结 self.UniUnet_model 的参数
        for param in self.UniUnet_model.parameters():
            param.requires_grad = False
        delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
        self.llama_model.load_state_dict(delta_ckpt, strict=False)
        self.device = torch.cuda.current_device()

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        

        if anomaly_embedding != None:
            # #print (f"bos_embeds:{bos_embeds.shape},p_before_embeds:{p_before_embeds.shape},img_embeds:{img_embeds.shape},p_middle_embeds:{p_middle_embeds.shape},p_after_embeds:{p_after_embeds.shape},anomaly_embedding:{anomaly_embedding.shape}")
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size()[1]+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size()[1]+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask
            #print (f"bos_embeds:{bos_embeds.shape},p_before_embeds:{p_before_embeds.shape},img_embeds:{img_embeds.shape},p_middle_embeds:{p_middle_embeds.shape},p_after_embeds:{p_after_embeds.shape},anomaly_embedding:{anomaly_embedding.shape}")
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size()[1]+p_middle_embeds.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+img_embeds.size()[1]+p_middle_embeds.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 


    def forward(self, images, texts):
        # anomalymap = inference(images, self.UniUnet_model, self.input_shape)
        # anomalymap = torch.sigmoid(anomalymap)   # shape [B, 1, D, H, W]
        # _,img_embeds1,img_embeds2,img_embeds3,img_embeds4 = self.UniUnet_model.encoder(images)
        self.UniUnet_model.eval()
        with torch.no_grad(): 
            img_embeds1,img_embeds2,img_embeds3,img_embeds4,anomalymap = self.UniUnet_model(images) 
        anomalymapscore = torch.sigmoid(anomalymap)
        img_embeds, anomaly_map_prompts = self.MultiScaleFusion(img_embeds1,img_embeds2,img_embeds3,img_embeds4,anomalymapscore)     # → [B, n, 2048]
        anomaly_map_prompts = anomaly_map_prompts.to(self.llama_model.dtype)
        img_embeds = img_embeds.to(self.llama_model.dtype)
        # print (f"anomaly_map_prompts:{anomaly_map_prompts.shape},img_embeds:{img_embeds.shape}")
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, texts, self.max_tgt_len)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask, anomaly_map_prompts)
        # inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)
        torch.cuda.empty_cache()
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss        
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        return loss, gen_acc, anomalymap

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        images = inputs['image_paths']
        # anomaly_maps = inference(images, self.UniUnet_model, self.input_shape) 
        # print(anomaly_maps)
        img_embeds1,img_embeds2,img_embeds3,img_embeds4,anomaly_maps = self.UniUnet_model(images)
        anomalymapscore = torch.sigmoid(anomaly_maps)

        feature_embeds, anomaly_map_prompts = self.MultiScaleFusion(img_embeds1,img_embeds2,img_embeds3,img_embeds4,anomalymapscore)     # → [B, n, 2048]
        feature_embeds = feature_embeds.to(self.llama_model.dtype)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        # self.prompt_learner.eval()
        anomaly_map_prompts = anomaly_map_prompts.to(self.llama_model.dtype)
        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        # inputs_embeds = torch.cat([bos_embeds, p_before_embeds, anomaly_map_prompts, p_after_embeds], dim=1) 
        # inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, p_after_embeds], dim=1)
        return inputs_embeds, anomaly_maps

    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
            }
        '''
        input_embeds, pixel_output = self.prepare_generation_embedding(inputs)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.llama_tokenizer.pad_token_id,
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        
        return output_text, pixel_output
    
