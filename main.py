def main(alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, t1, t2):
    # layers = torch.tensor([650, 150, 20])
    # layers = torch.tensor([512, 256, 80])
    layers = torch.tensor([120, 100, 80])
    dataset_name = 'block_ORL'
    
    model = DONMF_AE(dataset_name, 5, 1000, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, layers, t1= t1, t2= t2, loss_cal= False)
    model.pre_training()
    return model.training()