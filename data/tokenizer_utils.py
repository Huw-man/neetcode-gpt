from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        res = []
        for n in numbers:
            nt = []
            remainder = str(n)
            while len(remainder) > 0:
                for v in sorted(vocab, key=lambda x: -int(x)):
                    # print(remainder, v, remainder[:len(v)])
                    if remainder[:len(v)] == v:
                        nt.append(str(v))
                        remainder = remainder[len(v):]
                        break
                # print(remainder)
            res.append(nt)
        return res


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        nt = 0
        remainder = text
        while len(remainder) > 0:
            for v in sorted(vocab, key=lambda x: -len(x)):
                if remainder[:len(v)] == v:
                    nt +=1
                    remainder = remainder[len(v):]
                    break
        return nt


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(
            self.count_tokens(text, vocab) / len(text.split(' '))
            ,4)
