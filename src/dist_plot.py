import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

def dist_matplt(tensor,tup_shape,label,index):
    """
    Description : Plots the distribution of tensor
    args:
            tensor      - The tensor for which the distribution need to be plotted
            tup_shape   - Shape of the tensor in type tuple
            label       - Label in legend
            index       - Postion of the plot in subplot (max row*column)
    example:
        for name, parameter in model.named_parameters():
            if 'bias' not in name:
                dist_matplt(parameter,tuple(parameter.shape),name,index)
                dist_matplt(parameter.grad,tuple(parameter.grad.shape),"{}.grad".format(name),index+1)
                index = index + 2           
    """
    row = 7
    col = 5
    flatten = 1
    for i in tup_shape:
        flatten = flatten * i
    x = tensor.view(-1,flatten).detach().cpu().numpy()
    ax = fig.add_subplot(row, col, index)
    ax.hist(x[0],bins=50,label = label)
    plt.legend()
