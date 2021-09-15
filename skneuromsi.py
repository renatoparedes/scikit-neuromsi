import abc
import attr
import numpy as np

class MSIBrain(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __len__(self):
        """Number of modalities."""
        
    @abc.abstractmethod
    def __getitem__(self, modality):
        """"""
        
    @abc.abstractmethod
    def response(self):
        """"""


@attr.s
class ErnstBanks2002(MSIBrain):
    
    # Stimuli locations
    locs = attr.ib(default={"auditory": 0, "visual": 0})
    
    # Weights used to calculate the mean of the multimodal distribution
    weights = attr.ib(default={"auditory": 0.5, "visual": 0.5})
    
    # Variability of stimulus estimates
    e_sigmas = attr.ib(default={"auditory": 3, "visual": 4})

    # Result
    _result = attr.ib(factory=dict)

    # All possible locations
    pos_locs = attr.ib(default=np.arange(-20,20,0.01))

    def __attrs_post_init__(self):

        # From Alais and Burr 2004, the weights used to calculation the mean of the
        # multimodal distribution are:

        self.weights["auditory"] = self.e_sigmas["visual"]**2/(self.e_sigmas["auditory"]**2+self.e_sigmas["visual"]**2)
        self.weights["visual"] = self.e_sigmas["auditory"]**2/(self.e_sigmas["visual"]**2+self.e_sigmas["auditory"]**2)

        # From both Alais/Burr, 2004 and Ernst/Banks, 2002, the multisensory variability is:
        self.e_sigmas["multisensory"] = np.sqrt((self.e_sigmas["visual"]**2*self.e_sigmas["auditory"]**2)/(self.e_sigmas["auditory"]**2+self.e_sigmas["visual"]**2))

        # And the multisensory loc is:
        self.locs["multisensory"] = self.weights["visual"]*self.locs["visual"] + self.weights["auditory"]*self.locs["auditory"]

    def auditory_modality(self):
        """ 
        Computes auditory estimate 
        """

        if "auditiva" in self._result:
            raise ValueError()
        
        distr_a = (1/np.sqrt(2*np.pi*self.e_sigmas["auditory"]**2))*np.exp(-1*(((self.pos_locs-self.locs["auditory"])**2)/(2*self.e_sigmas["auditory"]**2)))

        self._result["auditory"] = distr_a
        
    def visual_modality(self):
        """ 
        Computes visual estimate
        """

        if "visual" in self._result:
            raise ValueError()
        
        distr_v = (1/np.sqrt(2*np.pi*self.e_sigmas["visual"]**2))*np.exp(-1*(((self.pos_locs-self.locs["visual"])**2)/(2*self.e_sigmas["visual"]**2)))

        self._result["visual"] = distr_v


    def multisensory_modality(self):
        """ 
        Computes multisensory estimate
        """

        if "multisensory" in self._result:
            raise ValueError()

        distr_m = (1/np.sqrt(2*np.pi*self.e_sigmas["multisensory"]**2))*np.exp(-1*(((self.pos_locs-self.locs["multisensory"])**2)/(2*self.e_sigmas["multisensory"]**2)))

        self._result["multisensory"] = distr_m

        
    def __len__(self):
        return 2
    
    def __getitem__(self, modality):
        if modality == "auditory":
            return self.auditory_modality
        elif modality == "visual":
            return self.visual_modality
        raise KeyError(modality)
        
    def response(self):
        return self._result["multisensory"]
    
    
ErnstBanks2002()


# %% Cue combination

# % We assume the visual and auditory probes can be displaced by up to 20
# % degrees of visual angle in either direction from the centre of the
# % display (as in Alais and Burr, 2004)
# x = -20:0.01:20; % possible locations

# x_v = 0;     % location for the visual stimulus
# sigma_v = 4;  % variability of visual location estimates
# x_a = 0;      % location for the auditory stimulus
# sigma_a = 3;  % variability of auditory location estimates

# % The distributions of visual and auditory location estimates are:
# distr_v = (1/sqrt(2*pi*sigma_v^2))*exp(-1*(((x-x_v).^2)/(2*sigma_v^2))); % visual
# distr_a = (1/sqrt(2*pi*sigma_a^2))*exp(-1*(((x-x_a).^2)/(2*sigma_a^2))); % auditory

# % From Alais and Burr 2004, the weights used to calculation the mean of the
# % multimodal distribution are:
# w_v = sigma_a^2/(sigma_v^2+sigma_a^2);
# w_a = sigma_v^2/(sigma_a^2+sigma_v^2);

# % And the mean itself is:
# x_va = w_v*x_v + w_a*x_a;

# % From both Alais/Burr, 2004 and Ernst/Banks, 2002, the variability is:
# sigma_va = sqrt((sigma_v^2*sigma_a^2)/(sigma_a^2+sigma_v^2));

# % Substituting into the standard equation for a Gaussian, the distribution is:
# distr_va = (1/sqrt(2*pi*sigma_va^2))*exp(-1*(((x-x_va).^2)/(2*sigma_va^2)));

# % To check it has the correct form, we compare it to the product of the
# % distributions:
# distr_prod = (1/(2*pi*sigma_v*sigma_a))*exp(-1*(((x-x_v).^2)/(2*sigma_v^2))-1*(((x-x_a).^2)/(2*sigma_a^2))); % calculate product (proportional to combined distribution)
# distr_prod = 100*distr_prod./sum(distr_prod); % normalize and rescale
# prodForPlot = distr_prod(1:100:4001); % extract a few values to plot as scatter

# % Plot all these distributions on the same set of axes
# distributions = figure;
# plot(x,distr_v,'b')
# hold on 
# plot(x,distr_a,'r')
# plot(x,distr_va,'m')
# scatter(-20:20,prodForPlot,'o','MarkerEdgeColor','k','MarkerFaceColor','g')
# %title('Distributions of location estimates','FontSize',14)
# ylabel('Probability density','FontSize',12)
# xlabel('Position (degrees of visual angle)','FontSize',12)
# legend('Visual','Auditory','Multimodal','Product of unimodals')
