import numpy

from njunmt.data.dataset import Dataset
from njunmt.data.text_inputter import ParallelTextInputter
from njunmt.data.text_inputter import TextLineInputter
from njunmt.data.vocab import Vocab
from njunmt.tests.data_iterator import EvalTextIterator
from njunmt.tests.data_iterator import TestTextIterator
from njunmt.tests.data_iterator import TrainTextIterator
from njunmt.utils.constants import Constants

vocab_src_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/vocab.zh"
vocab_trg_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/vocab.en"
codes_src_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/codes.src"
codes_trg_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/codes.trg"
vocab_srcbpe_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/vocab.zh.bpe"
vocab_trgbpe_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/vocab.en.bpe"
train_src_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/toy.zh"
train_trg_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/toy.en0"
eval_src_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/toy.zh"
eval_trg_file = "/Users/zhaocq/Documents/gitdownload/struct2struct/testdata/mt02.ref"


class TextInputterTest(object):
    def testTrainDataLoader(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_trg = Vocab(vocab_trg_file)
        vocab_srcbpe = Vocab(vocab_srcbpe_file, bpe_codes_file=codes_src_file)
        vocab_trgbpe = Vocab(vocab_trgbpe_file, bpe_codes_file=codes_trg_file)
        data = TrainTextIterator(
            train_src_file, train_trg_file,
            vocab_src, vocab_trg,
            batch_size=1)
        for (x, len_x), (y, len_y) in data:
            print (x[0])
            print (y[0])
            print (' '.join(vocab_src.convert_to_wordlist(x[0])))
            print (' '.join(vocab_trg.convert_to_wordlist(y[0])))
            break

    def testTestDataLoader(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_srcbpe = Vocab(vocab_srcbpe_file, bpe_codes_file=codes_src_file)
        data = TestTextIterator(
            train_src_file,
            vocab_srcbpe, batch_size=1)
        for x_str, (x, len_x) in data:
            print (x_str[0])
            print (x[0])
            print (' '.join(vocab_srcbpe.convert_to_wordlist(x[0])))
            break

    def testEvalDataLoader(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_trg = Vocab(vocab_trg_file)
        vocab_srcbpe = Vocab(vocab_srcbpe_file, bpe_codes_file=codes_src_file)
        vocab_trgbpe = Vocab(vocab_trgbpe_file, bpe_codes_file=codes_trg_file)
        data = EvalTextIterator(
            train_src_file, train_trg_file,
            vocab_src, vocab_trg,
            batch_size=1)
        for (x, len_x), (y, len_y) in data:
            print (x[0])
            print (' '.join(vocab_src.convert_to_wordlist(x[0])))
            print (y[0])
            print (' '.join(vocab_trg.convert_to_wordlist(y[0])))
            break

    def testParallelInputterTrain(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_trg = Vocab(vocab_trg_file)
        dataset = Dataset(vocab_src, vocab_trg,
                          train_src_file, train_trg_file,
                          eval_src_file, eval_trg_file)
        inputter = ParallelTextInputter(
            dataset,
            "train_features_file",
            "train_labels_file",
            batch_size=13,
            maximum_features_length=20,
            maximum_labels_length=20)

        inputter._cache_size = 10
        train_iter = TrainTextIterator(
            train_src_file, train_trg_file,
            vocab_src, vocab_trg,
            batch_size=13, maxlen_src=20, maxlen_trg=20)
        train_iter.k = 10
        input_fields = dataset.input_fields
        train_data = inputter.make_feeding_data()
        for a, b in zip(train_iter, train_data):
            x = a[0][0]
            x_len = a[0][1]
            y = a[1][0]
            y_len = a[1][1]
            x_new = b[1][input_fields[Constants.FEATURE_IDS_NAME]]
            x_len_new = b[1][input_fields[Constants.FEATURE_LENGTH_NAME]]
            y_new = b[1][input_fields[Constants.LABEL_IDS_NAME]]
            y_len_new = b[1][input_fields[Constants.LABEL_LENGTH_NAME]]
            assert x.all() == x_new.all()
            assert x_len.all() == x_len_new.all()
            assert y.all() == y_new.all()
            assert y_len.all() == y_len_new.all()
        print("Test Passed...")

    def testParallelInputterEval(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_trg = Vocab(vocab_trg_file)
        dataset = Dataset(vocab_src, vocab_trg,
                          train_src_file, train_trg_file,
                          eval_src_file, eval_trg_file)
        inputter = ParallelTextInputter(
            dataset,
            "eval_features_file",
            "eval_labels_file",
            batch_size=13,
            maximum_features_length=None,
            maximum_labels_length=None)

        eval_iter1 = EvalTextIterator(
            eval_src_file, eval_trg_file,
            vocab_src, vocab_trg,
            batch_size=13)

        eval_iter2 = TrainTextIterator(
            eval_src_file, eval_trg_file + "0",
            vocab_src, vocab_trg,
            batch_size=13, maxlen_src=1000, maxlen_trg=1000)
        input_fields = dataset.input_fields
        eval_data = inputter.make_feeding_data()
        for a, b, c in zip(eval_iter1, eval_iter2, eval_data):
            x1 = a[0][0]
            x_len1 = a[0][1]
            y1 = a[1][0]
            y_len1 = a[1][1]
            x2 = b[0][0]
            x_len2 = b[0][1]
            y2 = b[1][0]
            y_len2 = b[1][1]
            x_new = c[1][input_fields[Constants.FEATURE_IDS_NAME]]
            x_len_new = c[1][input_fields[Constants.FEATURE_LENGTH_NAME]]
            y_new = c[1][input_fields[Constants.LABEL_IDS_NAME]]
            y_len_new = c[1][input_fields[Constants.LABEL_LENGTH_NAME]]
            assert x1.all() == x_new.all() == x2.all()
            assert x_len1.all() == x_len_new.all() == x_len2.all()
            assert y1.all() == y_new.all() == y2.all()
            assert y_len1.all() == y_len_new.all() == y_len2.all()

        print("Test Passed...")

    def testTextInputterTest(self):
        vocab_src = Vocab(vocab_src_file)
        vocab_trg = Vocab(vocab_trg_file)
        dataset = Dataset(vocab_src, vocab_trg,
                          train_src_file, train_trg_file,
                          [eval_src_file], eval_trg_file)
        test_iter = TestTextIterator(
            train_src_file,
            vocab_src,
            batch_size=13)
        inputter = TextLineInputter(
            dataset,
            "eval_features_file",
            batch_size=13)
        input_fields = dataset.input_fields
        test_data = inputter.make_feeding_data()
        for a, b in zip(test_iter, test_data[0]):
            x_str = a[0]
            x = a[1][0]
            x_len = a[1][1]
            x_str_new = b[0]
            x_new = b[2][input_fields[Constants.FEATURE_IDS_NAME]]
            x_len_new = b[2][input_fields[Constants.FEATURE_LENGTH_NAME]]
            assert x.all() == x_new.all()
            assert x_len.all() == x_len_new.all()
            assert numpy.all([str1 == str2 for str1, str2 in zip(x_str, x_str_new)])
        print("Test Passed...")


if __name__ == "__main__":
    test = TextInputterTest()
    # test.testTrainDataLoader()
    # test.testTestDataLoader()
    # test.testParallelInputterTrain()
    # test.testParallelInputterEval()
    # test.testTextInputterTest()
