import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class RAGEvaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.bias_pipeline = pipeline("zero-shot-classification", model="Hate-speech-CNERG/dehatebert-mono-english")
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        self.toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert") 
      
    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer

    def evaluate_bleu_rouge(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_perplexity(self, text):
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score

    def evaluate_racial_bias(self, text):
        results = self.bias_pipeline([text], candidate_labels=["hate speech", "not hate speech"])
        bias_score = results[0]['scores'][results[0]['labels'].index('hate speech')]
        return bias_score

    def evaluate_meteor(self, candidates, references):
        nltk.download('punkt', quiet=True)  
        
        meteor_scores = [
            meteor_score([word_tokenize(ref)], word_tokenize(cand))
            for ref, cand in zip(references, candidates)
        ]
        return sum(meteor_scores) / len(meteor_scores)
    
    def evaluate_chrf(self, candidates, references):
        chrf_scores = [sentence_chrf(ref, cand) for ref, cand in zip(references, candidates)]
        return sum(chrf_scores) / len(chrf_scores)
    
    def evaluate_readability(self, text):
        flesch_ease = flesch_reading_ease(text)
        flesch_grade = flesch_kincaid_grade(text)
        return flesch_ease, flesch_grade

    def evaluate_semantic_similarity(self, candidate, reference):
        """
        Evaluates the semantic similarity between the candidate and reference 
        responses using Sentence Transformers.

        Args:
            candidate (str): The generated response.
            reference (str): The reference response.

        Returns:
            float: The cosine similarity score between the embeddings of the 
                   candidate and reference responses.
        """
        embeddings = self.sentence_transformer.encode([candidate, reference])
        return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]

    def evaluate_factual_consistency(self, candidate, reference, threshold=0.8):
        """
        Performs a simple factual consistency check by comparing the 
        semantic similarity of the candidate and reference responses to a 
        predefined threshold. 

        Note: This is a very basic implementation and can be improved 
        significantly with techniques like fact extraction and verification 
        against knowledge bases.

        Args:
            candidate (str): The generated response.
            reference (str): The reference response.
            threshold (float, optional): The similarity threshold above which 
                                        the responses are considered factually 
                                        consistent. Defaults to 0.8.

        Returns:
            bool: True if the responses are considered factually consistent, 
                  False otherwise. 
        """
        similarity = self.evaluate_semantic_similarity(candidate, reference)
        return similarity >= threshold

    def evaluate_question_relevance(self, question, response, threshold=0.7):
        """
        Evaluates the relevance of the generated response to the given 
        question using semantic similarity.

        Args:
            question (str): The input question.
            response (str): The generated response.
            threshold (float, optional): The similarity threshold above which 
                                        the response is considered relevant to 
                                        the question. Defaults to 0.7.

        Returns:
            bool: True if the response is considered relevant to the question,
                  False otherwise.
        """
        similarity = self.evaluate_semantic_similarity(question, response)
        return similarity >= threshold

    def evaluate_toxicity(self, text):
        """
        Evaluates the toxicity of the generated text.

        Args:
            text (str): The text to be evaluated.

        Returns:
            float: The toxicity score (higher means more toxic).
        """
        result = self.toxicity_pipeline(text)[0]
        return result['score'] if 'toxic' in result['label'] else 0.0

    def evaluate_context_relevance(self, context, response, threshold=0.7):
            """
            Evaluates the relevance of the generated response to the given
            context using semantic similarity.

            Args:
                context (str): The context text.
                response (str): The generated response.
                threshold (float, optional): The similarity threshold above which
                                            the response is considered relevant to
                                            the context. Defaults to 0.7.

            Returns:
                bool: True if the response is considered relevant to the context,
                    False otherwise.
            """
            similarity = self.evaluate_semantic_similarity(context, response)
            return similarity >= threshold

    def evaluate_answer_relevance(self, question, response, threshold=0.7):
        """
        Evaluates the relevance of the generated response to the given
        question using semantic similarity. This is similar to 
        `evaluate_question_relevance` but can be used when the question 
        is separate from the context.

        Args:
            question (str): The input question.
            response (str): The generated response.
            threshold (float, optional): The similarity threshold above which
                                        the response is considered relevant to
                                        the question. Defaults to 0.7.

        Returns:
            bool: True if the response is considered relevant to the question,
                  False otherwise.
        """
        similarity = self.evaluate_semantic_similarity(question, response)
        return similarity >= threshold

    
    def evaluate_all(self, question, response, reference):
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        racial_bias = self.evaluate_racial_bias(response)
        meteor = self.evaluate_meteor(candidates, references)
        chrf = self.evaluate_chrf(candidates, references)
        flesch_ease, flesch_grade = self.evaluate_readability(response)
        semantic_similarity = self.evaluate_semantic_similarity(response, reference)
        factual_consistency = self.evaluate_factual_consistency(response, reference)
        question_relevance = self.evaluate_question_relevance(question, response)
        context_relevance = self.evaluate_question_relevance(reference, response)
        answer_relevance = self.evaluate_question_relevance(question, response)
        toxicity = self.evaluate_toxicity(response) 

        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
            "Racial Bias": racial_bias,
            "METEOR": meteor,
            "CHRF": chrf,
            "Flesch Reading Ease": flesch_ease,
            "Flesch-Kincaid Grade": flesch_grade,
            "Semantic Similarity": semantic_similarity,
            "Factual Consistency": factual_consistency,
            "Question Relevance": question_relevance,
            "Context Relevance" : context_relevance,
            "Answer Relevance": answer_relevance,
             "Toxicity": toxicity
        }