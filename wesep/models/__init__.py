import wesep.models.bsrnn as bsrnn
import wesep.models.convtasnet as convtasnet
import wesep.models.dpccn as dpccn
import wesep.models.tfgridnet as tfgridnet
import wesep.modules.metric_gan.discriminator as discriminator
import wesep.models.bsrnn_multi_optim as bsrnn_multi

def get_model(model_name: str):
    if model_name.startswith("ConvTasNet"):
        return getattr(convtasnet, model_name)
    elif model_name.startswith("BSRNN_Multi"):
        return getattr(bsrnn_multi,model_name)
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
