from matplotlib import pyplot as plt
import numpy as np


scatter_size = 9
linewidth = 2


def _add_lines(vlines, hlines):
    for value, label, color in vlines:
        plt.axvline(x=value, label=label, color=color)
    for value, label, color in hlines:
        plt.axhline(y=value, label=label, color=color)


def plot_rk_curves(train_t, rk_data, figname=None):
    plt.plot(train_t, rk_data)
    plt.legend(["S", "I", "R"], shadow=True)
    plt.xlabel("Tempo (t)")
    plt.ylabel("Número de Indivíduos")
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_rk_data(train_t, rk_data, vlines=[], hlines=[], figname=None):
    plt.scatter(train_t, rk_data[:, 0], s=scatter_size, label="S", color="blue")
    plt.scatter(train_t, rk_data[:, 1], s=scatter_size, label="Dados I", color="red")
    plt.scatter(train_t, rk_data[:, 2], s=scatter_size, label="R", color="green")
    plt.xlabel("Tempo (t)")
    plt.ylabel("Número de Indivíduos")
    plt.grid()
    _add_lines(vlines, hlines)
    plt.legend()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_wdata(data_weight_hist, figname=None):
    plt.semilogy(np.arange(len(data_weight_hist)), data_weight_hist)
    plt.xlabel("Iteração")
    plt.ylabel(r"$\omega_{dados}$")
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show() 


def plot_losshistory(losshistory, figname=None):
    train = np.sum(losshistory.loss_train, axis=1)
    test = np.sum(losshistory.loss_test, axis=1)

    plt.semilogy(losshistory.steps, train, "-", label="Treinamento", linewidth=linewidth)
    # plt.semilogy(losshistory.steps, train, "o-", label="Treinamento", linewidth=2)
    # plt.semilogy(losshistory.steps, test, "x-", label="Teste", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro em escala logarítmica")

    plt.legend()
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_phys_losshistory(losshistory, n_physics=None, figname=None):
    loss_train = np.array(losshistory.loss_train) 
    train_phys = np.sum(loss_train[:,:loss_train.shape[1]], axis=1)

    plt.semilogy(losshistory.steps, train_phys, "-", label="Física", linewidth=linewidth, color="green")
    # plt.semilogy(losshistory.steps, train, "o-", label="Treinamento", linewidth=2)
    # plt.semilogy(losshistory.steps, test, "x-", label="Teste", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro em escala logarítmica")
    plt.legend()
    plt.grid()

    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    
    plt.show()


def plot_data_losshistory(losshistory, figname=None):
    loss_train = np.array(losshistory.loss_train) 
    train_dados = np.sum(loss_train[:,loss_train.shape[1]-1:], axis=1)

    plt.semilogy(losshistory.steps, train_dados, "-", label="Dados", linewidth=linewidth, color="orange")
    # plt.semilogy(losshistory.steps, train, "o-", label="Treinamento", linewidth=2)
    # plt.semilogy(losshistory.steps, test, "x-", label="Teste", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro em escala logarítmica")
    plt.legend()
    plt.grid()

    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_prevalence_data(data_t, I_data, figname=None):

    plt.scatter(data_t, I_data, label="Dados I", color="red", s=scatter_size)

    plt.xlabel("Tempo (t)")
    plt.ylabel("Número de Indivíduos")

    plt.legend()
    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_prevalence_results(I_data, data_t, I_pred, test_t, figname=None):

    plt.scatter(data_t, I_data, label="Dados I", color="red", s=scatter_size)
    plt.plot(test_t, I_pred, label="I previsto", linestyle="--", linewidth=linewidth, color="blue")

    plt.xlabel("Tempo (t)")
    plt.ylabel("Número de Indivíduos")

    plt.legend()
    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_results(sir_comparts, sir_data, sir_pred, data_t, N=1, vlines=[], hlines=[], figname=None):

    plt.plot(data_t, sir_comparts[:,0], label="S RK", color="blue", linewidth=linewidth)
    plt.plot(data_t, sir_comparts[:,1], label="I RK", color="red", linewidth=linewidth)
    plt.plot(data_t, sir_comparts[:,2], label="R RK", color="green", linewidth=linewidth)

    # plt.scatter(data_t, sir_data[:,0], label="Dados S", color="blue", s=scatter_size)
    plt.scatter(data_t, sir_data[:,1], label="Dados I", color="red", s=scatter_size)
    # plt.scatter(data_t, sir_data[:,2], label="Dados R", color="green", s=scatter_size)

    S_pred, I_pred = sir_pred[:, 0], sir_pred[:, 1]
    R_pred = N - S_pred - I_pred  

    plt.plot(data_t, sir_pred[:,0], label="S previsto", linestyle="--", color="blue", linewidth=linewidth)
    plt.plot(data_t, sir_pred[:,1], label="I previsto", linestyle="--", color="red", linewidth=linewidth)
    plt.plot(data_t, R_pred, label="R previsto", linestyle="--", color="green", linewidth=linewidth)

    _add_lines(vlines, hlines)

    plt.xlabel("Tempo (t)")
    plt.ylabel("Número de Indivíduos")

    plt.legend()
    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_beta_comparison(real_beta, pred_beta, test_t, vlines=[], hlines=[], figname=None):
    plt.rcParams['text.usetex'] = False
    
    plt.plot(test_t, real_beta, label=r"$\beta$ real", linewidth=linewidth)
    plt.plot(test_t, pred_beta, label=r"$\beta$ previsto", linestyle="--", linewidth=linewidth)
    
    plt.xlabel("Tempo (t)")
    plt.ylabel(r"$\beta$")
    
    for value, label, color in vlines:
        plt.axvline(x=value, label=label, color=color)

    for value, label, color in hlines:
        plt.axhline(y=value, label=label, color=color)
    
    plt.legend()
    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    
    plt.show()


def plot_beta_prediction(pred_beta, data_t, vlines=[], hlines=[], figname=None):
    plt.rcParams['text.usetex'] = False
    
    plt.plot(data_t, pred_beta, label=r"$\beta$ previsto", linestyle="--", linewidth=linewidth)
    
    plt.xlabel("Tempo (t)")
    plt.ylabel(r"$\beta$")
    
    _add_lines(vlines, hlines)
    
    plt.legend()
    plt.grid()

    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    
    plt.show()


def plot_correlation(
    pred_beta, 
    variable, 
    correlation_t,
    variable_label, 
    figname=None
):
    plt.rcParams['text.usetex'] = False
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Tempo (t)')
    ax1.set_ylabel(variable_label, color='tab:blue')
    ax1.bar(correlation_t, variable, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() 

    ax2.set_ylabel(r"$\beta$", color='tab:orange')
    ax2.plot(correlation_t, pred_beta, color='tab:orange', linewidth=linewidth)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.grid()
    fig.tight_layout()  

    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    
    plt.show()


def plot_beta_example(pred_beta, data_t, figname=None):
    plt.rcParams['text.usetex'] = False
    
    plt.plot(data_t, pred_beta, linewidth=linewidth)
    
    plt.xlabel("Tempo (t)")
    plt.ylabel(r"$\beta$")

    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    
    plt.show()


