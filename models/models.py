
def create_model(opt, _isTrain):
    model = None
    from .uprightnet_model import UprightNet
    model = UprightNet(opt, _isTrain)
    print("model [%s] was created" % (model.name()))
    return model
