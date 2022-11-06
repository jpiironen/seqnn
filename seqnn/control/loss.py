class PlanLoss:
    def __call__(self, trajectory: dict):
        raise NotImplementedError

    def get_relevant_tags(self):
        raise NotImplementedError

    def summarize_loss(self, losses, end_only=False, ignore_nans=False):
        assert losses.ndim == 3
        assert losses.shape[2] == 1
        if end_only:
            losses = losses[:, -1:, :]
        if ignore_nans:
            nan_mask = losses.isnan()
            losses[nan_mask] = 0.0
        return losses.mean(dim=1).squeeze()


class Setpoint(PlanLoss):
    def __init__(
        self,
        reference: dict,
        weights: dict = None,
        power: float = 2.0,
        end_only: bool = False,
    ):
        assert power > 0.0
        self.tags = list(reference.keys())
        self.reference = reference
        if weights is None:
            weights = {}
        for tag in self.tags:
            if tag not in weights:
                weights[tag] = 1.0
        self.weights = weights
        self.power = power
        self.end_only = end_only

    def get_relevant_tags(self):
        return self.tags

    def __call__(self, trajectory):
        loss = 0.0
        for tag, setpoint in self.reference.items():
            assert trajectory[tag].ndim == 3
            assert trajectory[tag].shape[2] == 1
            losses = (
                self.weights[tag] * (trajectory[tag] - setpoint).abs() ** self.power
            )
            loss += self.summarize_loss(losses, end_only=self.end_only)
        return loss
