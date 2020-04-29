import torch


class Compose(object):
    """ Use torchvision.transforms as a base.
    Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text):
        for t in self.transforms:
            text = t(text)
        return text

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``numpy.ndarray`` to tensor.
    """

    def __call__(self, arr):
        """
        Args:
            arr (numpy.ndarray): numpy ndarray to be converted to tensor.

        Returns:
            Tensor: Converted ndarray.
        """
        return torch.Tensor(arr)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BertTokenize(object):
    """Tokenize a text into input_ids and attention_mask by BERT
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        if max_length is not None and max_length > 512:
            self.max_length = 512
        else:
            self.max_length = max_length

    def __call__(self, text):
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             max_length=self.max_length,
                                             return_tensors='pt')

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        return input_ids[0], attention_mask[0]

    def __repr__(self):
        return self.__class__.__name__ + '()'
