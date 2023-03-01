"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_tensorboard.py
Description: A module for tensorboard
"""


from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger(object):
    def __init__(self, _log_dir, _opt=None):
        super(TensorBoardLogger, self).__init__()
        self.writer = SummaryWriter(log_dir=_log_dir)
        if _opt is not None:
            self.add_text(_tag='opt', _text_string=str(_opt), _step=None)

    def add_text(self, _tag, _text_string, _step):
        self.writer.add_text(tag=_tag, text_string=_text_string, global_step=_step)

    def add_scalar(self, _tag, _scalar_value, _step):
        # Usage: _logger.add_scalar( _tag='Loss/Train', _scalar_value=log_train[1], _step=epoch)
        self.writer.add_scalar(tag=_tag, scalar_value=_scalar_value, global_step=_step)

    def add_scalars(self, _main_tag, _tag_scalar_dict, _step):
        self.writer.add_scalars(main_tag=_main_tag, tag_scalar_dict=_tag_scalar_dict, global_step=_step)

    def add_image(self, _tag, _img_tensor, _step=None, _dataformats='CHW'):
        self.writer.add_image(tag=_tag, img_tensor=_img_tensor, global_step=_step, dataformats=_dataformats)

    def add_images(self, _tag, _img_tensor, _step=None, _dataformats='NCHW'):
        # The function, add_images has a color bug.
        # self.writer.add_images(tag=_tag, img_tensor=_img_tensor, global_step=_step, dataformats=_dataformats)
        for idx_batch in range(_img_tensor.shape[0]):
            suffix = '_' + str(idx_batch)
            self.add_image(_tag=(_tag+suffix), _img_tensor=_img_tensor[idx_batch], _step=_step, _dataformats=_dataformats[1:])

    def histo_summary(self, tag, values, step, bins=1000): # @Todo: To be implemented in the next version, 0.2.
        raise NotImplementedError
