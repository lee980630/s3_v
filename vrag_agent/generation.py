import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json
#generator 수정
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
import time as _time 
import random as _random 

# ▼▼▼[성능 측정 추가]▼▼▼ 수정
# GPUMonitor와 시간 기록을 위한 모듈을 가져옵니다.
from lsm_tmp.gpu_monitor import GPUMonitor
from datetime import datetime
# ▲▲▲[성능 측정 추가]▲▲▲


# ===== (1) DashScope 설정 =====
from http import HTTPStatus
from dotenv import load_dotenv

dotenv_dir = os.path.expanduser('~/workspace/VRAG_test/')

# 2. .env 파일의 전체 경로를 만듭니다.
dotenv_path = os.path.join(dotenv_dir, '.env')

# 3. 해당 경로의 .env 파일을 명시적으로 로드합니다.
load_dotenv(dotenv_path=dotenv_path)

try:
    import dashscope  # frozen generator (Qwen2.5-VL-72B 계열)
    import os as _os
    dashscope.base_http_api_url = _os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/api/v1"
    )
    _API_KEY = _os.getenv("DASHSCOPE_API_KEY") or _os.getenv("DASH_SCOPE_KEY")
    if not _API_KEY:
        raise RuntimeError("Set DASHSCOPE_API_KEY (or DASH_SCOPE_KEY).")
    dashscope.api_key = _API_KEY
    _HAS_DASHSCOPE = True
except Exception:
    _HAS_DASHSCOPE = False

# >>> ADDED: DashScope 멀티모달 헬퍼 (import 블록 바로 아래에 추가)
try:
    from dashscope import MultiModalConversation
except Exception:
    pass  # _HAS_DASHSCOPE=False 인 경우 대비

def _extract_text_from_multimodal(resp):
    """DashScope 멀티모달 응답에서 텍스트를 최대한 안전하게 추출"""
    try:
        ot = getattr(resp, "output_text", None)
        if ot:
            return str(ot).strip()
    except Exception:
        pass

    out = getattr(resp, "output", None)
    if not isinstance(out, dict):
        return None

    choices = out.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content") or []
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("text") is not None:
            texts.append(str(part["text"]))
    if texts:
        return "".join(texts).strip()

    if msg.get("text") is not None:
        return str(msg["text"]).strip()
    if out.get("text") is not None:
        return str(out["text"]).strip()
    return None


def _dashscope_call_with_fallback(model: str, messages: list, max_tokens: int):
    """SDK 버전 호환: max_output_tokens → 실패 시 max_tokens로 재시도"""
    try:
        return MultiModalConversation.call(
            model=model,
            messages=messages,
            max_output_tokens=max_tokens,
        )
    except TypeError:
        pass  # 일부 SDK는 max_output_tokens 미지원
    return MultiModalConversation.call(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

def _to_image_part(path: str) -> dict | None:
    """로컬 경로를 DashScope 이미지 파트(dict)로 변환 (file:// 스킴 강제)"""
    if not path:
        return None
    if not path.startswith("file://"):
        path = "file://" + os.path.abspath(path)
    return {"image": path}
# <<< ADDED 끝



def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int 
    num_gpus: int
    search_url: str = None
    #generator added
    crops_dir: str = "./agent_crops"
    frozen_model: str = "qwen2.5-vl-72b-instruct"   # Qwen2.5-VL-72B-Instruct 호환
    frozen_max_tokens: int = 1024
    generator_max_images: int = 8
    use_system_prompt: bool = True
    generator_batch_workers: int = 4
    frozen_max_retries: int = 3
    frozen_backoff_base: float = 1.5
    


class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))
        #generator added
        os.makedirs(self.config.crops_dir, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        self.cropped_images = None
        self.questions = None
                


    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def extract_tags(text):
            pattern = r"<(search|think|bbox|search_complete)>(.*?)</\1>" # generator 수정
            matches = re.findall(pattern, text, re.DOTALL)
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        merge_length = self.processor.image_processor.merge_size**2
        # print(self.retrievaled_images)
        for idx, obs_item in enumerate(next_obs):
            # invalid
            if isinstance(obs_item,str):
                next_obs_str.append(obs_item)
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # invalid
            elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.retrievaled_images[idx]) == 0:
                next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and <think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as you want. If you determine that no further knowledge is needed, you must finish with <search_complete>true</search_complete>. Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # crop
            elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
                try:
                    latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                    width, height = latest_image.size
                    raw_images_crop = Image.open(self.retrievaled_images[idx][-1])
                    raw_width, raw_height = raw_images_crop.size
                    if self.is_validation:
                        obs_item = [obs_item[0]-28, obs_item[1]-28, obs_item[2]+28, obs_item[3]+28]
                    crop_area = [int(raw_width * obs_item[0] / width), int(raw_height * obs_item[1] / height), int(raw_width * obs_item[2] / width), int(raw_height * obs_item[3] / height)]
                    crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                    input_images_list = [raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))]
                    raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                    #generator added
                    crop_path = os.path.join(self.config.crops_dir, f"{uuid.uuid4().hex}.jpg")
                    raw_images_list[0].save(crop_path)
                    self.cropped_images[idx].append(crop_path)
                    #                    

                    multi_modal_data.append({'image': raw_images_list})
                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']
                    obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                    raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                    obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)   
                except Exception as e:
                    next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as you want. If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_complete>. Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict())) 
            # ret image
            elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):
                img_file_list = [item['image_file'] for item in obs_item]
                for image_item in img_file_list:
                    if image_item not in self.retrievaled_images[idx]:
                        self.retrievaled_images[idx].append(image_item)
                        # input_images_list = img_file_list[:1]
                        input_images_list = [image_item]
                        break

                raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                multi_modal_data.append({'image': raw_images_list})
                image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                multi_modal_inputs.append(image_inputs)
                image_grid_thw = image_inputs['image_grid_thw']

                obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                obs_str = '\n<|im_start|>user\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                next_obs_str.append(obs_str)
            else:
                raise ValueError('invalid observation')
        
        next_obs_ids = self.tokenizer(
            next_obs_str, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():

            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            for idx, item in enumerate(next_obs_multi_modal_inputs):
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

            rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

        else:

            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                if len(multi_modal_data_item['image']) > 0:
                    # data
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']

        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], 151643, dtype=torch.int64)
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        

        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,151655) for row in raw_next_obs_ids]
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = 10240
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        meta_info = {}

        # ▼▼▼[성능 측정 추가] 1. 로그 파일 및 모니터 객체 초기화▼▼▼ 수정
        # 고유한 로그 파일 이름을 생성하여 모든 측정 결과를 한 파일에 기록합니다.
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"./logs/generation_detail_{current_time}_{uuid.uuid4().hex[:6]}.txt"
        
        # 측정 지점 1: 메인 모델(Actor)의 '계획' 생성 성능 측정용
        actor_monitor = GPUMonitor(log_file=log_filename, label="[1] Actor Generation (Planning)")
        
        # 측정 지점 2: 외부 도구(검색 API) 호출 시간 측정용
        tool_monitor = GPUMonitor(log_file=log_filename, label="[2] Tool Execution (Search API)")
        
        # 측정 지점 3: Frozen 모델의 '최종 답변' 생성 성능 측정용
        frozen_monitor = GPUMonitor(log_file=log_filename, label="[3] Frozen Generator (Answering)")
        # ▲▲▲[성능 측정 추가]▲▲▲        

        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}

        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        #generator added
        self.search_completed = [False] * gen_batch.batch['input_ids'].shape[0]

        # ===== (4) 첫 턴에서 질문 문자열 저장(원래 파싱 방식) & 컨테이너 준비 =====
        decoded_inputs = self.tokenizer.batch_decode(initial_input_ids, skip_special_tokens=True)
        '''
        최종 generator에게 초반 쿼리를 넘겨주기 위해서.
        '''
        self.questions = []
        for s in decoded_inputs:
            try:
                q = s.split('Question: ')[1].split(' \n\nassistant\n')[0]
            except Exception:
                q = s  # fallback
            self.questions.append(q)
        #


        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        self.cropped_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]      # generator added

        ############======================🚀Main generation loop🚀==================######################
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            ) #데이터 압축

            rollings = self._raw_prompt_ids(rollings)#전처리 

            active_mask = self.deactivate_batch(active_mask, rollings) #최대 길이를 넘으면 deactivate
            if not active_mask.sum():
                break
            
            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })                

            actor_monitor.start() #측정 지점 1: '계획' 생성 성능 측정 수정
            gen_output = self._generate_with_gpu_padding(rollings_active)
            actor_monitor.stop() #측정 끝

            meta_info = gen_output.meta_info     

            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            print(responses_str[0])

            
            # Execute in environment and process observations
            
            #개별 예제(example) 수준에서 빈자리를 채워주는(pad)'
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)


            #수정----#
            # 1. execute_predictions를 호출하기 전에 uids를 가져옵니다

            all_uids = rollings.non_tensor_batch['id']


            # 2. Execute in environment and process observations
            #    호출 시 uids를 두 번째 인자로 전달합니다.

            tool_monitor.start() #'행동'을 위한 외부 도구 호출 시간 측정▼▼▼ 수정
            next_obs, dones = self.execute_predictions(responses_str, all_uids, self.tokenizer.pad_token, active_mask)
            tool_monitor.stop() #측정 끝

            # --- 여기까지 ---

            #next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask) #수정 제거 uid 넘기기
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings)
            
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            
            # Update states            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids, #수정 제거 
                #padded_responses_ids, #수정 추가 uid
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids, #수정 제거 uid
                #padded_responses_ids, #수정 추가 uid
                next_obs_ids
            )



        # final LLM rollout
        if active_mask.sum():

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                all_uids = rollings.non_tensor_batch['id'] #수정 uid 추가 


                # # Execute in environment and process observations
                _, dones = self.execute_predictions( #ctive uid 추가 수정
                    responses_str, all_uids, self.tokenizer.pad_token, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # rollings.non_tensor_batch.pop('raw_prompt_ids')
        
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        ### check again
        
        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array
        # ===== generator added=====
        gen_to_tokenize = [""] * len(self.retrievaled_images)
        
        completed_indices = [i for i, flag in enumerate(self.search_completed) if flag]

        if completed_indices:
            batch_questions = []
            batch_paths = []
            
            for i in completed_indices:
                q = self.questions[i]
                paths = self._prepare_generator_images(self.retrievaled_images[i], self.cropped_images[i])
                batch_questions.append(q)
                batch_paths.append(paths)

            frozen_monitor.start()
            index2answer = self._call_frozen_generator_batch(
                completed_indices, batch_questions, batch_paths 
            )
            frozen_monitor.stop()

            for i in completed_indices:
                ans = index2answer.get(i, "")
                if ans:
                    gen_to_tokenize[i] = f"<answer>{ans}</answer>{self.tokenizer.eos_token}"

        ans_ids = self.tokenizer(
            gen_to_tokenize, padding='longest', return_tensors='pt', add_special_tokens=False
        )['input_ids']

        original_right_side = self._update_right_side(original_right_side, ans_ids)
        rollings = self._update_rolling_state(
            rollings, ans_ids, next_obs_ids=torch.zeros((ans_ids.shape[0], 0), dtype=torch.long)
        )
        #
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

# ... (generation.py 파일의 다른 부분은 모두 동일합니다) ...

    # execute_predictions 함수를 아래와 같이 수정합니다.
    def execute_predictions(self, predictions: List[str], uids: np.ndarray, pad_token: str, active_mask=None, do_search=True) -> List[str]:
        cur_actions, contents = self.postprocess_predictions(predictions)  

        next_obs, dones = [], []
        
        bbox_list = [content for action, content in zip(cur_actions, contents) if action == 'bbox']
        
        search_requests = []
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if action == 'search':
                m = re.search(r'(\d+)$', str(uids[i]))
                search_id = int(m.group(1)) if m else -1
                
                search_requests.append({
                    "query": content,
                    "id": str(search_id),
                    "request_idx": i  
                })                   

        if do_search:
            if len(search_requests) > 0:              
                batch_size = 100
                search_results_list = []
                for i in range(0, len(search_requests), batch_size):
                    batch_reqs = search_requests[i:i + batch_size]
                    response = requests.post(self.config.search_url, json=batch_reqs)                    
                    search_results_single_batch = response.json()
                    search_results_list.extend(search_results_single_batch)                  

                results_map = {item['request_idx']: item.get('results', []) for item in search_results_list}
                assert len(results_map) == len(search_requests)
            else:
                results_map = {}
        else:
            results_map = {}
         

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'search':
                    result_for_this_agent = results_map.get(i, [])
                    next_obs.append(result_for_this_agent)
                    dones.append(0)
                elif action == 'bbox':
                    try:
                        bbox_value = json.loads(bbox_list.pop(0))
                        if len(bbox_value) == 4 and bbox_value[0] >= 0 and bbox_value[1] >= 0 and bbox_value[2] >= 0 and bbox_value[3] >= 0:
                            next_obs.append(bbox_value)
                        else:
                            raise ValueError("Invalid bbox value")
                    except:
                        next_obs.append('\n<|im_start|>user\nYour previous action is invalid. \n The bbox format is invalid. Expected format: JSON array [x1, y1, x2, y2] with all values >= 0. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                elif action == 'search_complete':
                    is_true = contents[i].strip().lower() == 'true'
                    if is_true:
                        self.search_completed[i] = True
                    next_obs.append('')
                    dones.append(1)  # trajectory 종료
                else:
                    next_obs.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine using <search> query </search> and the user will return the search results. Whenever you retrieve an image, you may crop it for a clearer view using <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as you want. If you determine that no further external knowledge is needed, you must finish with <search_complete>true</search_compelte>. Otherwise, continue with <search> or <bbox> actions until you are ready to finish. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
        
        # 모든 결과를 소비했는지 최종 확인
        # assert len(search_results) == 0 # 이 로직은 더 이상 유효하지 않으므로 제거합니다.

        return next_obs, dones


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|bbox|search_complete)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    #generator added
    # ===== (8) generator 이미지 준비 =====
    def _prepare_generator_images(self, originals: List[str], crops: List[str]) -> List[str]:
        # 존재하는 파일만, 중복 제거, 최대 장수 제한
        seen = set()
        out = []
        for p in (originals + crops):
            if p and (p not in seen) and os.path.exists(p):
                seen.add(p)
                out.append(p)
            if len(out) >= self.config.generator_max_images:
                break
        return out



    def _call_frozen_generator_single(self, question: str, image_paths: List[str]) -> Tuple[int, str]:
        if not _HAS_DASHSCOPE:
            return (0, "")

        try:
            # 빈 프롬프트 방지(400 회피)
            qtext = (question or "").strip() or "."

            sys_prompt = (
                "You are a visual QA generator. "
                "Use only the provided images and the user question. "
                "Return ONLY the final answer text without extra explanations."
            )

            # 이미지 파트 구성 (file:// 강제)
            user_content = []
            if image_paths:
                for p in image_paths:
                    part = _to_image_part(p)  # >>> ADDED: helper 사용
                    if part:
                        user_content.append(part)
            user_content.append({"text": f"Question: {qtext}"})

            messages = []
            if getattr(self.config, "use_system_prompt", True):
                messages.append({"role": "system", "content": [{"text": sys_prompt}]})
            messages.append({"role": "user", "content": user_content})

            try:
                resp = _dashscope_call_with_fallback(
                    model=self.config.frozen_model,
                    messages=messages,
                    max_tokens=int(getattr(self.config, "frozen_max_tokens", 256)),
                )
            except Exception:
                return (0, "")

            code = getattr(resp, "status_code", None)
            if code == HTTPStatus.OK:
                text = _extract_text_from_multimodal(resp) or ""
                return (200, text)
            
            return (int(code) if isinstance(code, HTTPStatus) else (code or 0), "")
        except Exception:
            return (0, "")


    def _call_frozen_generator_batch(
        self,
        indices: List[int],
        questions: List[str],
        images_list: List[List[str]],
    ) -> Dict[int, str]:

        results: Dict[int, str] = {}
        if not indices:
            return results
        

        workers = max(1, int(getattr(self.config, "generator_batch_workers", 4)))
        workers = min(workers, 4)
        max_retries = int(getattr(self.config, "frozen_max_retries", 3))
        backoff_base = float(getattr(self.config, "frozen_backoff_base", 1.5))

        def _once_with_retry(idx: int, q: str, paths: List[str]) -> Tuple[int, str]:
            delay = 0.0
            for attempt in range(max_retries):
                if delay > 0:
                    _time.sleep(delay)
                code, ans = self._call_frozen_generator_single(q, paths)
                
                if code == 200:
                    if ans:
                        return idx, ans 
                    else:
                        return idx, ""
                
                if code in (429, 500, 502, 503, 504, 0):
                    delay = (backoff_base ** attempt) + _random.uniform(0, 0.2)
                    continue 
                
                return idx, ""
            
            return idx, ""

        for start in range(0, len(indices), workers):
            end = start + workers 
            chunk_idx = indices[start:end]
            chunk_q = questions[start:end]
            chunk_img = images_list[start:end]

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_once_with_retry, i, q, p) for i, q, p in zip(chunk_idx, chunk_q, chunk_img)]
                for f in as_completed(futs):
                    try:
                        i, ans = f.result()
                    except Exception:
                        i, ans = None, ""
                    if i is not None:
                        results[i] = ans or ""

            _time.sleep(0.05)
        
        return results 



