from .decoder import Decoder
from .decoder_splatting_cuda_2dgs import DecoderSplattingCUDACfg
from .decoder_splatting_cuda_2dgs import DecoderSplattingCUDA as DecoderSplattingCUDA2DGS

DECODERS = {
    # "splatting_cuda": DecoderSplattingCUDA,
    "splatting_cuda_2dgs": DecoderSplattingCUDA2DGS,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
