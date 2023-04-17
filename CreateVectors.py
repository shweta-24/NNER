class Vectors:

    STOPWORDS = ["of", "the", "and", "in", "to", "a", "that", "by", "with", "is", "was", "for", "as", "not", "from", "an", "or", "are", "on", "we", "in", "this", "these", "be", "but"]

    def __init__(self):

        self.__vectors = []

    # Creates vectors from a corpus (represented as a string array, with each sentence as one elem)
    def create_vectors(self, data, ngram):

        for row in data:

            row = row.split()

            first_word = True

            prev_word = ""

            for i in range(len(row)-ngram+1):


                if ngram == 1:
                    self.__vectors.append(self.get_vector(row[i], first_word, prev_word))
                else:
                    sub = row[i:i+ngram]
                    self.__vectors.append(self.get_multiword_vector(sub, first_word, prev_word))


                first_word = False
                prev_word = row[i].lower()

    # Creates a vector for a word
    def get_vector(self, word, first_word, prev_word):

        vector = [0] * 6

        # Is the first letter a capital letter?
        if word[0].isupper():
            vector[0] = 1

        # Is it the first word of a sentence
        if first_word:
            vector[1] = 1


        for char_idx in range(len(word)):

            # Is any letter in the word (other than the first) a capital letter?
            if char_idx > 0 and word[char_idx].isupper():
                vector[2] = 1


            # Does it include a numeral?
            # TODO maybe check that is doesn't ONLY include numerals?
            if word[char_idx].isdigit():
                vector[3] = 1


            # Does it include special characters?
            if word[char_idx] in ['-', '/']:
                vector[4] = 1

        # Was the previous word a particle?
        if prev_word in ["a", "an", "the"]:
            vector[5] = 1

            
        return vector

    def get_multiword_vector(self, wordarray, first_word, prev_word):

        vector = []

        has_stopword = False

        # Appends the feature vectors for each individual word
        for word in wordarray:
            vector += self.get_vector(word, first_word, prev_word)
            first_word = False
            prev_word = word

            if word.lower() in self.STOPWORDS:
                has_stopword = True
        

        # Is any of the words a stop word?
        if has_stopword:
            vector.append(1)
        else:
            vector.append(0)
        
        return vector
        



    # Writes the created vectors to a file
    def write_vectors_to_file(self, filepath):

        with open(filepath, 'w') as file:

            for vector in self.__vectors:
                for elem in vector:
                    file.write(str(elem))
                    file.write(' ')
                file.write('\n')



