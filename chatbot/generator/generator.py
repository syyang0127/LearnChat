from transformers import pipeline
from typing import Optional

class Generator:
    def __init__(self) -> None:
        """
        Initialize the generator with a pre-trained model for text generation.
        Using facebook/opt-125m as it's a relatively small but capable model.
        """
        self.generator = pipeline(
            "text-generation",
            model="facebook/opt-125m",
            device=-1,  # -1 for CPU, 0 for GPU if available
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        
    def generate(self, prompt: str) -> str:
        """
        Generate natural continuation of the given prompt.
        
        Args:
            prompt (str): The input text to continue
            
        Returns:
            str: Generated continuation of the prompt
            
        Example:
            prompt: "I want"
            return: " to be a doctor"
        """
        try:
            # Generate the continuation
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 10,  # Limit the length of generation
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # Extract only the generated part (without the prompt)
            generated_text = result[0]['generated_text']
            continuation = generated_text[len(prompt):]
            
            return continuation
            
        except Exception as e:
            print(f"Error in text generation: {str(e)}")
            return " [Error in generation]"
    
    def _clean_response(self, text: str) -> str:
        """
        Clean the generated response by removing any unwanted artifacts.
        
        Args:
            text (str): Raw generated text
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove any special tokens that might have been generated
        text = text.replace('<|endoftext|>', '')
        text = text.replace('<|pad|>', '')
        
        return text.strip()