import random
import re
import numpy as np
import math
import itertools
from sklearn import decomposition
import editdistance


from typing import List, Tuple, Iterable

# This file stores the "solutions" the bot had intended,
# when you play as agent and the bot as spymaster.
log_file = open("log_file", "w")

N_WORDS = 4 # Words uses in total
K_WORDS = 3 # Words shown in each clue
N_ROUNDS = 8
HARDNESS = 2 # 1 is hardest, N_ROUNDS is easiest

class Reader:
    def read_picks(
        self, words: List[str], my_words: Iterable[str], cnt: int
    ) -> List[str]:
        picks = []
        while len(picks) < cnt:
            guess = None
            while guess not in words:
                guess = input("Your guess: ").strip().lower()
            picks.append(guess)
            if guess in my_words:
                print("Correct!")
            else:
                print("Wrong :(")
                break
        return picks

    def read_word(self, label, word_set) -> Tuple[str, int]:
        while True:
            clue = input(f'{label} (e.g. car): ').lower().strip()
            if clue not in word_set:
                print("I don't understand that word.")
                continue
            return clue

    def print_words(self, words: List[str], nrows: int):
        longest = max(map(len, words))
        print()
        for row in zip(*(iter(words),) * nrows):
            for word in row:
                print(word.rjust(longest), end=" ")
            print()
        print()

    def yesno(self, question):
        while True:
            response = input(f'{question} [y/n]: ')
            if response in ('y', 'yes'):
                return True
            if response in ('n', 'no'):
                return False

    def read_reg(self, label, reg):
        while True:
            response = input(label)
            match = re.search(reg, response)
            if match:
                return match.groups()

class Codenames:
    def __init__(self):
        self.vectors = np.array([])
        self.word_list = []
        self.weirdness = []
        self.word_to_index = {}
        self.codenames = []

    def load(self, datadir):
        # Glove word vectors
        print("...Loading vectors")
        self.vectors = np.load(f"{datadir}/glove.6B.300d.npy")
        dim = self.vectors[0].shape[0]

        self.vectors -= self.vectors.mean(axis=1, keepdims=True)
        # TODO: Is this the right number of components to remove?
        pca = decomposition.PCA(n_components=10)
        pca.fit(self.vectors)
        proj = pca.transform(self.vectors)
        self.vectors -= proj @ pca.components_
        # TODO: Maybe try just permuting the coordinates?
        self.vectors = self.vectors @ np.random.randn(dim, dim)

        # List of all glove words
        print("...Loading words")
        self.word_list = [w.lower().strip() for w in open(f"{datadir}/words")]
        # TODO: Make weirdness factor more easily adjustable
        self.weirdness = np.array([math.log(i + 100) for i in range(len(self.word_list))])

        # Indexing back from word to indices
        print("...Making word to index dict")
        self.word_to_index = {w: i for i, w in enumerate(self.word_list)}

        # All words that are allowed to go onto the table
        print("...Loading codenames")
        self.codenames: List[str] = [
            word
            for word in (w.lower().strip().replace(" ", "-") for w in open("wordlist2"))
            if word in self.word_to_index
        ]

        print("Ready!")

    def word_to_vector(self, word: str) -> np.ndarray:
        """
        :param word: To be vectorized.
        :return: The vector.
        """
        return self.vectors[self.word_to_index[word]]

    def most_similar_to_given(self, clue: str, choices: List[str]) -> str:
        """
        :param clue: Clue from the spymaster.
        :param choices: Choices on the table.
        :return: Which choice to go for.
        """
        clue_vector = self.word_to_vector(clue)
        return max(choices, key=lambda w: self.word_to_vector(w) @ clue_vector)

    def code_list(self):
        res = list(itertools.permutations(range(N_WORDS), K_WORDS))
        random.shuffle(res)
        return res

    def play_spymaster(self, reader: Reader):
        """
        Play a complete game, with the robot giving clues.
        """
        if reader.yesno('Should I pick the words myself?'):
            words = random.sample(self.codenames, N_WORDS)
            print('Ok.')
        else:
            print('Which four words should I make cluses for?')
            words = []
            for c in code:
                words.append(reader.read_word(f'Word {c+1}', self.word_to_index.keys()))
            print('Ok.')
        print()
        print('My words are:', ', '.join(f'{w} {i+1}' for i, w in enumerate(words)))


        dim = self.vectors[0].shape[0]
        blocklen = dim * HARDNESS // N_ROUNDS

        # Move secret words out of the array
        to_remove  = [self.word_to_index[word] for word in words]
        word_vectors = self.vectors[to_remove]
        legal_clues = np.delete(self.vectors, to_remove, 0)
        legal_clue_words = np.delete(self.word_list, to_remove, 0)
        weirdness = np.delete(self.weirdness, to_remove, 0)
        #from scipy import stats
        #print(stats.describe(weirdness))

        used = set()
        for i, code in enumerate(self.code_list()):
            print(f'\nRound {i+1}')
            #block = legal_clues[:, i*blocklen : (i+1)*blocklen]
            block = legal_clues.take(range(i*blocklen, (i+1)*blocklen), axis=1, mode='wrap')
            clues = []
            ws = []
            for c in code:
                while True:
                    clue_piece = word_vectors[c].take(range(i*blocklen , (i+1)*blocklen), mode='wrap')
                    row = np.argmax(block @ clue_piece / (1 + weirdness))
                    word = legal_clue_words[row]
                    # Remove annoying numbers and symbols
                    if not word.isalpha() \
                            or len(word) <= 2 \
                            or words[c] in word \
                            or any(w in word for w in used) \
                            or weirdness[row] > 11:
                            #or editdistance.distance(word, words[c]) <= 1 \
                        # print('Bad:', word)
                        block[row] *= 0
                        continue
                    clues.append(word)
                    ws.append(weirdness[row])
                    used.add(word)
                    break

            print('Clues:', ' '.join(clues), '('+' '.join(f'{int(w)}' for w in ws)+')')
            guess = tuple(int(g)-1 for g in reader.read_reg('Your guesses: ', r'(\d) (\d) (\d)'))
            if code == guess:
                print('Correct!')
            else:
                print('The correct order was:', ', '.join(f'{w} {c+1}' for c, w in zip(code, clues)))



    def play_agent(self, reader: Reader):
        """
        Play a complete game, with the robot trying to guess.
        """
        if reader.yesno('Do you need inspiration for words?'):
            print('Ok. Your four words are:')
            words = random.sample(self.codenames, N_WORDS)
            print(' '.join(f'({i+1}) {w}' for i, w in enumerate(words)))
        else:
            print('Ok.')

        known = [] # code -> [word]
        used = set()
        for i in range(N_ROUNDS):
            while True:
                code = random.sample(range(4), 3)
                if code not in used:
                    used.add(code)
                    break
            print('Give clues for', '.'.join('{c+1}' for c in code))
            clues = []
            for c in code:
                clue = reader.read_word(f'for {c+1}', self.word_to_index.keys())
                # TODO


def main():
    cn = Codenames()
    cn.load("dataset")
    reader = Reader()
    while True:
        try:
            mode = input("\nWill you guess or give clues?: ")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        try:
            if mode == "guess":
                cn.play_spymaster(reader)
            elif mode == "give clues":
                cn.play_agent(reader)
        except KeyboardInterrupt:
            # Catch interrupts from play functions
            pass


main()
