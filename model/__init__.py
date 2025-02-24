from .vid2seq import _get_tokenizer, Vid2Seq
from .texttitling import TextTilingTokenizer

def build_vid2seq_model(args, tokenizer):
    model = Vid2Seq(t5_path=args.model_name,
                    num_features=args.max_feats,
                    video_embedding_dim=args.embedding_dim,
                    depth=args.depth,
                    num_heads=args.heads,
                    mlp_dim=args.mlp_dim,
                    visual_dropout=args.visual_encoder_dropout,
                    encoder_dropout=args.text_encoder_dropout,
                    decoder_dropout=args.text_decoder_dropout,
                    tokenizer=tokenizer,
                    num_bins=args.num_bins,
                    label_smoothing=args.label_smoothing,
                    use_speech=args.use_speech,
                    use_video=args.use_video)
    return model