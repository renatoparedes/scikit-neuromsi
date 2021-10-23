# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000943
# https://github.com/multisensoryperceptionlab/BCIT
# https://github.com/multisensoryperceptionlab/BCIT/blob/master/BCIM%20ToolBox/bciModel.m

from . import core


@core.neural_msi_model
class Kording2007:

    # hiperparameters
    # TODO look up for default values
    # 1 =  auditory, 2 = visual

    auditory_sigma = core.hparameter()
    auditory_var = core.hparameter()

    @auditory_var.default
    def _auditory_var_default(self):
        return self.auditory_var ** 2

    visual_sigma = core.hparameter()
    visual_var = core.hparameter()

    @visual_var.default
    def _visual_var_default(self):
        return self.visual_var ** 2

    # auditory_loc = core.hparameter()
    # visual_loc = core.hparameter()

    # internals

    p_common = core.internal(default=0.5)
    prior_sigma = core.internal()
    prior_var = core.internal()

    @prior_var.default
    def _prior_var_default(self):
        return self.prior_sigma ** 2

    prior_mu = core.internal()

    multisensory_com_var = core.internal()

    @multisensory_com_var.default
    def _multisensory_com_var_default(self):
        return (
            self.auditory_var * self.visual_var
            + self.auditory_var * self.prior_var
            + self.visual_var * self.prior_var
        )

    auditory_ind_var = core.internal()

    @auditory_ind_var.default
    def _auditory_ind_var_default(self):
        return self.auditory_var + self.prior_var

    visual_ind_var = core.internal()

    @visual_ind_var.default
    def _visual_ind_var_default(self):
        return self.visual_var + self.prior_var

    multisensory_var_hat = core.internal()

    @multisensory_var_hat.default
    def _multisensory_var_hat_default(self):
        return 1 / (
            1 / self.auditory_var + 1 / self.visual_var + 1 / self.prior_var
        )

    auditory_var_hat = core.internal()

    @auditory_var_hat.default
    def _auditory_var_hat_default(self):
        return 1 / (1 / self.auditory_var + 1 / self.prior_var)

    visual_var_hat = core.internal()

    @visual_var_hat.default
    def _visual_var_hat_default(self):
        return 1 / (1 / self.visual_var + 1 / self.prior_var)


# x1 = bsxfun(@plus, stim1, sig1 * randn(N,1));
# x2 = bsxfun(@plus, stim2, sig2 * randn(N,1));
