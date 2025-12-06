from matplotlib import pyplot as plt


def plot_rk_curves(train_t, rk_data, figname):
    plt.plot(train_t, rk_data)
    plt.legend(["S", "I", "R"], shadow=True)
    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")
    plt.grid()
    plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_rk_data(train_t, rk_data, figname):
    plt.scatter(train_t, rk_data[:, 0], s=2.5)
    plt.scatter(train_t, rk_data[:, 1], s=2.5)
    plt.scatter(train_t, rk_data[:, 2], s=2.5)
    plt.legend(["S", "I", "R"], shadow=True)
    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")
    plt.grid()
    plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_wdata(data_weight_hist, figname):
    plt.semilogy(np.arange(len(data_weight_hist)), data_weight_hist)
    plt.xlabel("Iteração")
    plt.ylabel("$\omega_{dados}$")
    plt.savefig(f"../../images/{figname}.png")
    plt.show() 


def plot_training(losshistory, figname):
    train = np.sum(losshistory.loss_train, axis=1)
    test = np.sum(losshistory.loss_test, axis=1)

    plt.semilogy(losshistory.steps, train, "-", label="Treinamento", linewidth=1)
    # plt.semilogy(losshistory.steps, train, "o-", label="Treinamento", linewidth=2)
    # plt.semilogy(losshistory.steps, test, "x-", label="Teste", linewidth=2)

    plt.xlabel("Iteração")
    plt.ylabel("Erro em escala logarítmica")

    plt.legend()
    plt.grid()
    plt.savefig(f"../../images/{figname}.png")
    plt.show()


def plot_results(sir_data, figname):
    plt.rcParams['text.usetex'] = False
    nop_test = 100

    test_t = np.linspace(t0, tf, nop_test).reshape(-1, 1)
    y_pred = model.predict(test_t)

    S_pred, I_pred = y_pred[:, 0], y_pred[:, 1]
    R_pred = 1 - S_pred - I_pred  

    sir_test = sir_sol.sol(test_t.reshape(nop_test)).T

    S_real = sir_test[:,0]
    I_real = sir_test[:,1]
    R_real = sir_test[:,2]

    plt.plot(train_t, S_real, label="S RK", color="blue")
    plt.plot(train_t, I_real, label="I RK", color="red")
    plt.plot(train_t, R_real, label="R RK", color="green")

    plt.scatter(train_t, sir_data[:,0], label="Dados S", color="blue", s=2.5)
    plt.scatter(train_t, sir_data[:,1], label="Dados I", color="red", s=2.5)
    plt.scatter(train_t, sir_data[:,2], label="Dados R", color="green", s=2.5)

    plt.plot(test_t, S_pred, label="S previsto", linestyle="--", color="blue")
    plt.plot(test_t, I_pred, label="I previsto", linestyle="--", color="red")
    plt.plot(test_t, R_pred, label="R previsto", linestyle="--", color="green")

    plt.xlabel("Tempo (t)")
    plt.ylabel("Fração da População")

    plt.legend()
    plt.grid()
    plt.savefig(f"../../images/{figname}.png")

    plt.show()


def plot_beta(beta_t, figname):
    beta_pred = y_pred[:, 2]
    beta_real = beta_t(test_t)
    plt.plot(test_t, beta_real, label=r"$\beta$ real")
    plt.plot(test_t, beta_pred, label=r"$\beta$ previsto", linestyle="--")
    plt.xlabel("Tempo (t)")
    plt.ylabel(r"$\beta$")
    plt.legend()
    plt.grid()
    plt.savefig(f"../../images/{figname}.png")
    plt.plot()