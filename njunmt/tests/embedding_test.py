import tensorflow as tf

from njunmt.tests.embedding import WordEmbedder


class EmbeddingTest(tf.test.TestCase):
    def testWordEmbedder(self):
        embedder = WordEmbedder(20, 10)
        words1 = [[1, 3, 5, 7], [3, 4, 7, 9]]
        words2 = [1, 3]  # time = 0
        words3 = [5, 7]  # time = 2
        emb1 = embedder.embed_words(words1)
        emb2 = embedder.embed_words(words2, 0)
        emb3 = embedder.embed_words(words3, 2)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            emb1 = sess.run(emb1)
            emb2 = sess.run(emb2)
            emb3 = sess.run(emb3)
            emb_table = sess.run(embedder._embeddings)
            self.assertAllEqual(emb1[:, 0, :], emb2)
            self.assertAllEqual(emb2, emb_table[words2])
            self.assertAllEqual(emb1[:, 2, :], emb3)
            self.assertAllEqual(emb3, emb_table[words3])

    def testWordEmbedderSin(self):
        embedder = WordEmbedder(20, 10, timing="sinusoids")
        words1 = [[1, 3, 5, 7], [3, 4, 7, 9]]
        words2 = [1, 3]  # time = 0
        words3 = [5, 7]  # time = 2
        emb1 = embedder.embed_words(words1)
        emb2 = embedder.embed_words(words2, 0)
        emb3 = embedder.embed_words(words3, 2)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            emb1 = sess.run(emb1)
            emb2 = sess.run(emb2)
            emb3 = sess.run(emb3)
            self.assertAllEqual(emb1[:, 0, :], emb2)
            self.assertAllEqual(emb1[:, 2, :], emb3)

    def testWordEmbedderEmb(self):
        embedder = WordEmbedder(20, 10, timing="emb")
        words1 = [[1, 3, 5, 7], [3, 4, 7, 9]]
        words2 = [1, 3]  # time = 0
        words3 = [5, 7]  # time = 2
        emb1 = embedder.embed_words(words1)
        emb2 = embedder.embed_words(words2, 0)
        emb3 = embedder.embed_words(words3, 2)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            emb1 = sess.run(emb1)
            emb2 = sess.run(emb2)
            emb3 = sess.run(emb3)
            self.assertAllEqual(emb1[:, 0, :], emb2)
            self.assertAllEqual(emb1[:, 2, :], emb3)

if __name__ == "__main__":
    tf.test.main()
