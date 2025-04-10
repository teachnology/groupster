from .util import diversity, diversity_cost, restriction_cost


class Group:
    """Group class.

    This class is used to represent a group of individuals in a cohort. It contains
    methods to compute the diversity of the group and costs of the group.

    Parameters
    ----------
    data : pd.DataFrame
        The group data to compute the diversity or cost of.
    bools : list, optional
        A list of boolean columns in the data. These are categorical characteristics.
    nums : list, optional
        A list of numeric columns in the data. These are continuous values.

    """

    def __init__(self, data, bools=None, nums=None):
        self.data = data
        self.bools = bools
        self.nums = nums

    @property
    def diversity(self):
        """Compute the diversity of the group.

        For more details, see the `util.diversity` function.
        """
        return diversity(data=self.data, bools=self.bools, nums=self.nums)

    def diversity_cost(self, cohort_diversity, cost_fn=None):
        """Compute the diversity cost of the group.

        This method computes the cost of the group diversity compared to the cohort
        diversity. The cost is computed using the ``cost_fn`` function.

        The default cost function is `util.diversity_cost`. For details on the default
        cost function, see the `util.diversity_cost` function.

        Parameters
        ----------
        cohort_diversity : pd.Series
            The diversity of the cohort.
        cost_fn : callable, optional
            A function to compute the group cost. The default is `util.diversity_cost`.

        Returns
        -------
        float
            The diversity cost of the group.

        """
        if cost_fn is None:
            cost_fn = diversity_cost

        return cost_fn(
            cohort_diversity=cohort_diversity,
            group_diversity=self.diversity,
        )

    def restriction_cost(
        self, keep_together=None, keep_separate=None, bool_min=None, cost_fn=None
    ):
        """Compute the restriction cost of the group.

        This method computes the restriction cost of the group. The cost is computed
        using the ``cost_fn`` function. The default cost function is
        `util.restriction_cost`. For details on the default cost function, see the
        `util.restriction_cost` function.

        Parameters
        ----------
        keep_together : list of list of str, optional
            A list of lists of indices in ``data`` that should be kept together.
        keep_separate : list of list of str, optional
            A list of lists of indices in ``data`` that should be kept separate.
        bool_min : float, optional
            The minimum number of people with the boolean characteristic in the group.
            For example, if we want at least two females in each group, we can set
            ``bool_min={'female': 2}``.
        cost_fn : callable, optional
            A function to compute the restriction cost of a group. The default is
            `util.restriction_cost`.

        Returns
        -------
        float
            The restriction cost of the group.

        """
        if cost_fn is None:
            cost_fn = restriction_cost

        return cost_fn(self.data, keep_together, keep_separate, bool_min)
