from matplotlib import pyplot as plt
import numpy as np


def plot_rk_curves(train_t, rk_data, figname=None):
    plt.plot(train_t, rk_data)
    plt.legend(["S", "I", "R"], shadow=True)
    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_rk_data(train_t, rk_data, figname=None):
    plt.scatter(train_t, rk_data[:, 0], s=2.5)
    plt.scatter(train_t, rk_data[:, 1], s=2.5)
    plt.scatter(train_t, rk_data[:, 2], s=2.5)
    plt.legend(["S", "I", "R"], shadow=True)
    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_wdata(data_weight_hist, figname=None):
    plt.semilogy(np.arange(len(data_weight_hist)), data_weight_hist)
    plt.xlabel("Iteração")
    plt.ylabel("$\omega_{dados}$")
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show() 


def plot_losshistory(losshistory, figname=None):
    train = np.sum(losshistory.loss_train, axis=1)
    test = np.sum(losshistory.loss_test, axis=1)

    plt.semilogy(losshistory.steps, train, "-", label="Treinamento", linewidth=1)
    # plt.semilogy(losshistory.steps, train, "o-", label="Treinamento", linewidth=2)
    # plt.semilogy(losshistory.steps, test, "x-", label="Teste", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro em escala logarítmica")

    plt.legend()
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_results(sir_comparts, sir_data, sir_pred, data_t, figname=None):
    plt.rcParams['text.usetex'] = False
    nop_test = 100

    # S_pred, I_pred = y_pred[:, 0], y_pred[:, 1]
    # R_pred = 1 - S_pred - I_pred  

    plt.plot(data_t, sir_comparts[:,0], label="S RK", color="blue")
    plt.plot(data_t, sir_comparts[:,1], label="I RK", color="red")
    plt.plot(data_t, sir_comparts[:,2], label="R RK", color="green")

    plt.scatter(data_t, sir_data[:,0], label="Dados S", color="blue", s=2.5)
    plt.scatter(data_t, sir_data[:,1], label="Dados I", color="red", s=2.5)
    plt.scatter(data_t, sir_data[:,2], label="Dados R", color="green", s=2.5)

    plt.plot(data_t, sir_pred[:,0], label="S previsto", linestyle="--", color="blue")
    plt.plot(data_t, sir_pred[:,1], label="I previsto", linestyle="--", color="red")
    plt.plot(data_t, sir_pred[:,2], label="R previsto", linestyle="--", color="green")

    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")

    plt.legend()
    plt.grid()
    
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_beta(beta_t, pred_beta, data_t, figname=None):
    real_beta = beta_t(data_t)
    plt.plot(data_t, real_beta, label=r"$\beta$ real")
    plt.plot(data_t, pred_beta, label=r"$\beta$ previsto", linestyle="--")
    plt.xlabel("Tempo (t)")
    plt.ylabel(r"$\beta$")
    plt.legend()
    plt.grid()
    if figname is not None:
        plt.savefig(f"../../images/{figname}.png")
    plt.plot()