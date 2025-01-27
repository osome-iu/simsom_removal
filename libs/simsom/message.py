"""
Class modeling a message. 
A message has 2 intrinsic, independent attributes:
- Quality following an exponential distribution (estimated using empirical data)
- Appeal, following a right-skewed distribution
"""

import random
import numpy as np
import warnings
from scipy.stats import beta


class Message:
    def __init__(
        self,
        id: int,
        user_id: int,
        is_by_bot=False,
        author_class="normal",
        phi=0,
        quality_distr=None,
        is_shadow=False,
        get_legality=False,
    ) -> None:
        """
        Initializes an instance for a message.
        Quality and appeal values are decided by the parameter phi.
        Parameters:
            - id (int): unique identifier for this message
            - is_by_bot (int): 1 if the message is by bot, else 0
            - phi (float): range [0,1].
            If phi=0, there is no appeal advantage to messages by bot. Meaning appeal of bot and human messages drawn from the same distribution
            - quality_distr: contains the information to calculate quality (following an exponential distribution if there is none, a beta-like distribution
            distribution if it contains data)
            - is_shadow (bool): flag used to indicate whether the content comes from a banned agent or not
        """

        self.id = id
        self.user_id = user_id
        self.author_class = author_class
        self.is_by_bot = is_by_bot
        self.is_shadow = is_shadow
        self.quality_distr = quality_distr
        self.phi = phi
        self.get_legality = get_legality
        self.set_quality_appeal(self.quality_distr)
        self.set_legality(self.quality_distr)

    def expon_quality(self, lambda_quality=-5) -> float:
        """
        Return a quality value x via inverse transform sampling
        Pdf of quality: $f(x) \sim Ce^{-\lambda x}$, 0<=x<=1
        $C = \frac{\lambda}{1-e^{-\lambda}}$
        """
        x = random.random()
        return np.log(1 - x + x * np.e ** (-1 * lambda_quality)) / (-1 * lambda_quality)

    def custom_beta_quality(self, distribution_param: str, round: bool = True) -> float:
        """
        Return a quality value x via beta distribution with alpha and beta params
        Since we use a custom beta distribution version, we have limits within which we want our values to come out.
        If, for example, values between 0 and 0.3 are set, then we discard values greater than 0.3 and take the first one that falls within the set range.
        """
        if distribution_param:
            params = eval(distribution_param)
            alpha, beta, lower, upper = params
            checked = False
            # we can truncate values as we need
            while not checked:
                if round:
                    quality = round(np.random.beta(alpha, beta), 2)
                else:
                    quality = np.random.beta(alpha, beta)
                if quality >= lower and quality <= upper:
                    checked = True
            return quality
        else:
            # if we do not pass any param or None value we use exponential quality
            return self.expon_quality()

    def appeal_func(self, exponent=5) -> float:
        """
        Return an appeal value a following a right-skewed distribution via inverse transform sampling
        Pdf of appeal: $P(a) = (1+\alpha)(1-a)^{\alpha}$
        exponent = alpha+1 characterizes the rarity of high appeal values --- the larger alpha, the more skewed the distribution
        """

        # if the users that post the message are under shadowban the appeal should be 0 to not be reshared
        if self.is_shadow:
            return 0
        else:
            u = random.random()
            return 1 - (1 - u) ** (1 / exponent)

    def set_legality(self, quality_distr: str) -> None:
        # Get the legality of the message
        # which class does the user belong to?
        # if illegal, return the probability that the content is illegal
        if self.get_legality:
            if not self.quality_distr:
                raise ValueError(
                    "quality_distr (probability a message is illegal) must be set if get_legality is True"
                )
            prob_illegal = self.custom_beta_quality(
                distribution_param=quality_distr, round=False
            )
            if random.random() < prob_illegal:
                self.legality = "illegal"
            else:
                self.legality = "legal"
        else:
            self.legality = "legal"
        return None

    def set_quality_appeal(self, quality_distr: str) -> None:
        """
        Set message attributes: quality, appeal
        Quality, appeal drawn via inverse transform sampling from 2 distinct distributions https://en.wikipedia.org/wiki/Inverse_transform_sampling
        Note that the 2 random numbers generated below may or may not include 1, see https://docs.python.org/3/library/random.html#random.uniform.
            - For systems with bots, bot message is fixed to have quality=0, so we don't need to worry about it.
            - For systems without bots, we use beta_quality with alpha and beta params, we do not have "bot" so "bad" content are not related to users anymore
            since we use life_time for moderation we should move here the code for illegal contents
            '''
        """

        if self.get_legality:
            # quality doesn't matter in this case
            self.quality = -1
            # appeal is the same for all users
            self.appeal = self.appeal_func()
        else:
            # appeal value of a "normal" message by humans (in modeling systems with bots)
            human_appeal = self.appeal_func()
            u = random.random()  # random number to decides appeal
            if self.is_by_bot:
                self.quality = 0
                self.appeal = 1 if u < self.phi else human_appeal
            else:
                self.quality = self.custom_beta_quality(
                    distribution_param=quality_distr
                )
                self.appeal = human_appeal

        return
