import wesep.models.bsrnn as bsrnn
import wesep.models.convtasnet as convtasnet
import wesep.models.dpccn as dpccn
import wesep.models.tfgridnet as tfgridnet


def get_model(model_name: str):
    if model_name.startswith("ConvTasNet"):
        return getattr(convtasnet, model_name)
    elif model_name.startswith("BSRNN"):
        return getattr(bsrnn, model_name)
    elif model_name.startswith("DPCNN"):
        return getattr(dpccn, model_name)
    elif model_name.startswith("TFGridNet"):
        return getattr(tfgridnet, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)


if __name__ == "__main__":
    print(get_model("ConvTasNet"))
