import wesep.models.aed_kws_asr_phone as aed_kws_asr_phone
import wesep.models.kwts_encoder as kwts_encoder

# TSE models with heavy dependencies (silero_vad, s3prl, etc.) are lazy-loaded.
# They are only needed when training TSE models, not for KCE.
# Keep the import attempts in try/except so KCE models work standalone.
try:
    import wesep.models.bsrnn as bsrnn
except ImportError:
    bsrnn = None
try:
    import wesep.models.convtasnet as convtasnet
except ImportError:
    convtasnet = None
try:
    import wesep.models.dpccn as dpccn
except ImportError:
    dpccn = None
try:
    import wesep.models.tfgridnet as tfgridnet
except ImportError:
    tfgridnet = None
try:
    import wesep.modules.metric_gan.discriminator as discriminator
except ImportError:
    discriminator = None
try:
    import wesep.models.bsrnn_multi_optim as bsrnn_multi
except ImportError:
    bsrnn_multi = None
try:
    import wesep.models.bsrnn_feats as bsrnn_feats
except ImportError:
    bsrnn_feats = None


def get_model(model_name: str):
    if model_name.startswith("AEDKWSASRPhone"):
        return getattr(aed_kws_asr_phone, model_name)
    elif model_name.startswith("KWTSEncoder"):
        return getattr(kwts_encoder, model_name)
    elif model_name.startswith("ConvTasNet"):
        return getattr(convtasnet, model_name)
    elif model_name.startswith("BSRNN_Multi"):
        return getattr(bsrnn_multi, model_name)
    elif model_name.startswith("BSRNN_Feats"):
        return getattr(bsrnn_feats, model_name)
    elif model_name.startswith("BSRNN"):
        return getattr(bsrnn, model_name)
    elif model_name.startswith("DPCCN"):
        return getattr(dpccn, model_name)
    elif model_name.startswith("TFGridNet"):
        return getattr(tfgridnet, model_name)
    elif model_name.startswith("CMGAN"):
        return getattr(discriminator, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
