from typing import List
import heapq


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus_chars = list(corpus)
        cc = corpus_chars
        merges = []
        for _ in range(num_merges):
            freq = {}
            for i in range(len(cc)-1):
                ca, cb = cc[i], cc[i+1]
                if (ca, cb) not in freq:
                    freq[(ca, cb)] = 0
                freq[(ca, cb)] +=1
            print(freq)
            top_pair = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[0]
            print(top_pair)
            merges.append(list(top_pair[0]))
            merged_token = top_pair[0][0] + top_pair[0][1]
            

            next_c = [cc[0]]
            i = 1
            print(next_c)
            while i < len(cc):
                ca, cb = next_c[-1], cc[i]
                if merged_token == ca+cb:
                    next_c.pop(-1)
                    next_c.append(merged_token)
                else:
                    next_c.append(cb)
                i += 1
            print(next_c)
            cc = next_c
        return merges