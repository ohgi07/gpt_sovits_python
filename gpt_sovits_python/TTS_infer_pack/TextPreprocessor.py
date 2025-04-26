import os, sys

from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
import LangSegment
from gpt_sovits_python.text import chinese
from typing import Dict, List, Tuple
from gpt_sovits_python.text import cleaned_text_to_sequence as text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from gpt_sovits_python.TTS_infer_pack.text_segmentation_method import split_big_text, get_method as get_seg_method
import logging
from tools.i18n.i18n import I18nAuto, scan_language_list
from gpt_sovits_python.text import symbols as symbols_v1
from gpt_sovits_python.text import symbols2 as symbols_v2
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

def clean_special(text, language, special_s, target_symbol, version=None):
    if version is None:version=os.environ.get('version', 'v2')
    if version == "v1":
        symbols = symbols_v1.symbols
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols
        language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean","yue":"cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("gpt_sovits_python.text."+language_module_map[language],fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text

def clean_text(text, language, version=None):
    if version is None:
        version = os.environ.get('version', 'v2')
    if version == "v1":
        symbols = symbols_v1.symbols  # Make sure this is a valid list
        language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}
    else:
        symbols = symbols_v2.symbols  # Also a valid list
        language_module_map = {
            "zh": "chinese2",
            "ja": "japanese",
            "en": "english",
            "ko": "korean",
            "yue": "cantonese"
        }

    if language not in language_module_map:
        language = "en"
        text = " "

    # Check special cases
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol, version)
    
    # Import the correct language module
    language_module = __import__(
        "gpt_sovits_python.text." + language_module_map[language],
        fromlist=[language_module_map[language]]
    )
    
    # Normalize text if function available
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text

    # Actual g2p logic
    if language in ("zh", "yue"):
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None

    # Replace unknown symbols
    phones = ['UNK' if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts:str, threshold:int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result





class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM, 
                 tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
        self.punctuation = {'!', '?', '…', ',', '.', '-', " "}

    def segment_and_extract_feature_for_text(self, text: str, language: str, version: str = "v1") -> Tuple[list, torch.Tensor, str]:
            """Process text segment and extract features"""
            try:
                return self.get_phones_and_bert(text, language, version)
            except Exception as e:
                logger.error(f"Error in segment_and_extract_feature_for_text: {str(e)}")
                return None, None, ""
            
    
    def clean_text_inf(self, text: str, language: str, version: str = "v2") -> Tuple[list, list, str]:
        """Clean and process text with proper module handling"""
        try:
            # Ensure text is a string and not empty
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                return [1], [1], ""

            # Direct call to text_cleaner function
            try:
                phones, word2ph, norm_text = clean_text(text, language, version)
            except Exception as e:
                logger.error(f"Error in text_cleaner: {str(e)}")
                return [1], [1], text

            # Convert phones to sequence
            try:
                if isinstance(phones, (list, tuple)):
                    phone_sequence = text_to_sequence(phones, version)
                else:
                    logger.error(f"Invalid phones type: {type(phones)}")
                    return [1], [1], text
            except Exception as e:
                logger.error(f"Error in text_to_sequence: {str(e)}")
                return [1], [1], text

            # Validate outputs
            if not isinstance(word2ph, list):
                word2ph = [1]
            if not isinstance(norm_text, str):
                norm_text = text

            return phone_sequence, word2ph, norm_text

        except Exception as e:
            logger.error(f"Error in clean_text_inf: {str(e)}")
            return [1], [1], text

    def _process_single_language(self, text: str, language: str, version: str) -> Tuple[list, torch.Tensor, str]:
        """Process single language text with proper module handling"""
        try:
            language = language.replace("all_", "")
            
            if language == "en":
                try:
                    LangSegment.setfilters(["en"])
                    segments = list(LangSegment.getTexts(text))
                    formattext = " ".join(seg["text"] for seg in segments if isinstance(seg, dict) and "text" in seg)
                except Exception as e:
                    logger.warning(f"Error in LangSegment processing: {e}")
                    formattext = text
            else:
                formattext = text

            formattext = formattext.replace("  ", " ").strip()
            
            if language == "zh" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "zh", version)

            phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
            bert = self._get_bert_features(language, phones, word2ph, norm_text)
            
            return phones, bert, norm_text

        except Exception as e:
            logger.error(f"Error in _process_single_language: {str(e)}")
            return [1], torch.zeros((1024, 1), dtype=torch.float32).to(self.device), text

    def get_phones_and_bert(self, text: str, language: str, version: str, final: bool = False) -> Tuple[list, torch.Tensor, str]:
        """Get phones and BERT features with proper module handling"""
        try:
            # Direct language processing
            if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
                result = self._process_single_language(text, language, version)
            elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
                result = self._process_multi_language(text, language, version)
            else:
                result = self._process_single_language(text, language, version)

            # Validate result
            phones, bert, norm_text = result
            if not final and len(phones) < 6:
                return self.get_phones_and_bert("." + text, language, version, final=True)
                
            return phones, bert, norm_text

        except Exception as e:
            logger.error(f"Error in get_phones_and_bert: {str(e)}")
            return [1], torch.zeros((1024, 1), dtype=torch.float32).to(self.device), text


    def _process_multi_language(self, text: str, language: str, version: str) -> Tuple[list, torch.Tensor, str]:
        """Process text in multi-language mode with fixed LangSegment handling"""
        try:
            textlist = []
            langlist = []
            
            try:
                LangSegment.setfilters(["zh", "ja", "en", "ko"])
                segments = list(LangSegment.getTexts(text))
            except Exception as e:
                logger.warning(f"Error in LangSegment processing: {e}, falling back to single language")
                return self._process_single_language(text, language, version)

            if not segments:
                logger.warning("No segments found, falling back to single language")
                return self._process_single_language(text, language, version)

            for segment in segments:
                if not isinstance(segment, dict) or "lang" not in segment or "text" not in segment:
                    continue
                    
                if language == "auto":
                    langlist.append(segment["lang"])
                elif language == "auto_yue":
                    langlist.append("yue" if segment["lang"] == "zh" else segment["lang"])
                else:
                    langlist.append("en" if segment["lang"] == "en" else language)
                textlist.append(segment["text"])

            if not textlist:
                logger.warning("No valid text segments found, falling back to single language")
                return self._process_single_language(text, language, version)

            phones_list = []
            bert_list = []
            norm_text_list = []
            
            for i in range(len(textlist)):
                try:
                    phones, word2ph, norm_text = self.clean_text_inf(textlist[i], langlist[i], version)
                    bert = self._get_bert_features(langlist[i], phones, word2ph, norm_text)
                    
                    phones_list.append(phones)
                    norm_text_list.append(norm_text)
                    bert_list.append(bert)
                except Exception as e:
                    logger.warning(f"Error processing segment {i}: {e}")
                    continue

            if not phones_list:
                logger.warning("No segments processed successfully, falling back to single language")
                return self._process_single_language(text, language, version)

            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
            
            return phones, bert, norm_text
            
        except Exception as e:
            logger.error(f"Error in _process_multi_language: {str(e)}")
            return self._process_single_language(text, language, version)

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        """Extract BERT features with dimension validation"""
        try:
            if not text or not word2ph:
                raise ValueError("Empty text or word2ph")

            # Ensure text is properly formatted
            text = text.strip()
            if not text:
                raise ValueError("Empty text after stripping")

            with torch.no_grad():
                # Tokenize and move to device
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get BERT features
                res = self.bert_model(**inputs, output_hidden_states=True)
                hidden_states = res["hidden_states"][-3:-2]
                
                # Ensure we have hidden states
                if not hidden_states:
                    raise ValueError("No hidden states produced")
                
                res = torch.cat(hidden_states, -1)[0].cpu()[1:-1]

            # Validate dimensions
            if len(word2ph) != len(text):
                logger.warning(f"Length mismatch: word2ph={len(word2ph)}, text={len(text)}")
                # Adjust word2ph if necessary
                word2ph = [1] * len(text)

            # Create phone-level features
            phone_level_feature = []
            for i in range(len(word2ph)):
                if i < res.shape[0]:  # Ensure we don't exceed tensor dimensions
                    repeat_feature = res[i].repeat(word2ph[i], 1)
                    phone_level_feature.append(repeat_feature)

            if not phone_level_feature:
                raise ValueError("No phone-level features generated")

            # Concatenate and validate final tensor
            final_feature = torch.cat(phone_level_feature, dim=0).T
            
            # Ensure minimum dimension
            if final_feature.shape[1] == 0:
                logger.warning("Empty feature tensor, creating dummy tensor")
                final_feature = torch.zeros((1024, 1), dtype=torch.float32)

            return final_feature.to(self.device)

        except Exception as e:
            logger.error(f"Error in BERT feature extraction: {str(e)}")
            # Return a valid tensor with minimum dimensions
            return torch.zeros((1024, max(1, len(word2ph))), dtype=torch.float32).to(self.device)

    def _get_bert_features(
        self, 
        language: str, 
        phones: list, 
        word2ph: list, 
        norm_text: str
    ) -> torch.Tensor:
        """Get BERT features with enhanced validation"""
        try:
            language = language.replace("all_", "")
            if language == "zh":
                features = self.get_bert_feature(norm_text, word2ph)
                # Ensure minimum dimension
                if features.shape[1] == 0:
                    features = torch.zeros((1024, len(phones)), dtype=torch.float32)
            else:
                features = torch.zeros((1024, len(phones)), dtype=torch.float32)

            return features.to(self.device)

        except Exception as e:
            logger.error(f"Error in _get_bert_features: {str(e)}")
            # Ensure we return a valid tensor
            return torch.zeros((1024, max(1, len(phones))), dtype=torch.float32).to(self.device)

     
    def preprocess(self, text: str, lang: str, text_split_method: str, version: str = "v2") -> List[Dict]:
        """Main preprocessing pipeline with enhanced error handling"""
        try:
            logger.info("Starting text preprocessing")
            text = self.replace_consecutive_punctuation(text)
            texts = self.pre_seg_text(text, lang, text_split_method)
            
            if not texts:
                logger.warning("No valid text segments produced")
                return []

            result = []
            logger.info("Extracting BERT features")
            
            for text_segment in tqdm(texts):
                try:
                    phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(
                        text_segment, lang, version
                    )
                    
                    if phones is None or not norm_text:
                        logger.warning(f"Invalid output for segment: {text_segment[:50]}...")
                        continue

                    if not isinstance(bert_features, torch.Tensor):
                        logger.error(f"Invalid BERT features type: {type(bert_features)}")
                        continue

                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    result.append(res)
                    
                except Exception as e:
                    logger.error(f"Error processing segment '{text_segment[:50]}...': {str(e)}")
                    continue
                    
            return result
            
        except Exception as e:
            logger.error(f"Fatal error in preprocessing: {str(e)}")
            return []

    def pre_seg_text(self, text: str, lang: str, text_split_method: str) -> List[str]:
        """Enhanced text segmentation with validation"""
        try:
            text = text.strip("\n")
            if not text:
                return []

            # Add initial punctuation if needed
            if text[0] not in self.splits and len(self.get_first(text)) < 4:
                text = "。" + text if lang != "en" else "." + text

            logger.info(f"Input text: {text}")
            
            # Get segmentation method
            seg_method = get_seg_method(text_split_method)
            text = seg_method(text)
            
            # Clean and split text
            text = text.replace("\n\n", "\n")
            _texts = text.split("\n")
            _texts = self.filter_text(_texts)
            _texts = merge_short_text_in_array(_texts, 5)
            
            texts = []
            for segment in _texts:
                segment = segment.strip()
                if not segment or not re.sub(r"\W+", "", segment):
                    continue
                    
                # Add final punctuation if needed
                if segment[-1] not in self.splits:
                    segment += "。" if lang != "en" else "."
                
                # Handle long segments
                if len(segment) > 510:
                    texts.extend(split_big_text(segment))
                else:
                    texts.append(segment)
                    
            logger.info(f"Segmented texts: {texts}")
            return texts
            
        except Exception as e:
            logger.error(f"Error in text segmentation: {str(e)}")
            return []

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        """Extract BERT features with validation"""
        try:
            if not text or not word2ph:
                raise ValueError("Empty text or word2ph")
                
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                res = self.bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
                
            if len(word2ph) != len(text):
                raise ValueError(f"Length mismatch: word2ph={len(word2ph)}, text={len(text)}")
                
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)
                
            return torch.cat(phone_level_feature, dim=0).T
            
        except Exception as e:
            logger.error(f"Error in BERT feature extraction: {str(e)}")
            return torch.zeros((1024, len(word2ph)), dtype=torch.float32).to(self.device)

    def get_first(self, text: str) -> str:
        """Extract first segment of text"""
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        return re.split(pattern, text)[0].strip()

    def filter_text(self, texts: List[str]) -> List[str]:
        """Filter valid text segments"""
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError("No valid text segments found")
            
        return [text for text in texts if text not in [None, " ", ""]]

    def replace_consecutive_punctuation(self, text: str) -> str:
        """Normalize consecutive punctuation"""
        punctuations = ''.join(re.escape(p) for p in self.punctuation)
        pattern = f'([{punctuations}])([{punctuations}])+'
        return re.sub(pattern, r'\1', text)