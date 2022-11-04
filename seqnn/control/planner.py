import torch


class AdamPlanner:
    def __init__(self, model, data_past, control_plan, control_lims, lr=0.1):
        self.model = model
        self.past = data_past
        self.plan = control_plan
        # this will be held fixed
        self.plan_init = {
            key: tensor.requires_grad_(False) for key, tensor in control_plan.items()
        }
        self.tags_to_optimize = list(control_lims.keys())
        self.optim_masks = self.create_optim_masks(self.tags_to_optimize)
        self.lims_min, self.lims_max = self.create_group_limits(control_lims)

        # this will contain the tensors that are being optimized (optimize in the unconstrained space)
        self.plan_param_uncons = {
            key: torch.zeros_like(tensor).add_(tensor).requires_grad_(True)
            for key, tensor in self.to_unconstrained(control_plan).items()
        }
        self.optimizer = torch.optim.Adam(
            [p for p in self.plan_param_uncons.values()],
            lr=lr,
        )

    def create_group_limits(self, tag_lims):
        lims_min = {}
        lims_max = {}
        for group, tensor in self.plan_init.items():
            lims_min[group] = torch.zeros(1, 1, tensor.shape[2])
            lims_max[group] = torch.ones(1, 1, tensor.shape[2])
        for tag, (min, max) in tag_lims.items():
            group, index = self.model.get_group_and_index(tag)
            lims_min[group][:, :, index] = min
            lims_max[group][:, :, index] = max
        return lims_min, lims_max

    def create_optim_masks(self, tags_to_optimize):
        # create boolean masks indicating which elements in the dictionary of tensors (=plan) shall be optimized
        optim_masks = {}
        for group, tensor in self.plan_init.items():
            optim_masks[group] = torch.zeros_like(tensor, dtype=bool)
        for tag in tags_to_optimize:
            group, index = self.model.get_group_and_index(tag)
            optim_masks[group][:, :, index] = True
        return optim_masks

    def to_constrained(self, plan):
        return {
            key: torch.sigmoid(x) * (self.lims_max[key] - self.lims_min[key])
            + self.lims_min[key]
            for key, x in plan.items()
        }

    def to_unconstrained(self, plan):
        return {
            key: torch.logit(
                (x - self.lims_min[key]) / (self.lims_max[key] - self.lims_min[key])
            )
            for key, x in plan.items()
        }

    def get_plan_for_pred(self):
        # this will return a plan dictionary with gradients enabled, but so that
        # those controls are being masked out that should not be optimized
        plan_param = self.to_constrained(self.plan_param_uncons)
        plan = {
            group: torch.where(self.optim_masks[group], param, self.plan_init[group])
            for group, param in plan_param.items()
        }
        return plan

    def get_loss(self, plan):
        pred = self.model.predict(self.past, plan)
        # TODO: IMPLEMENT HERE THE LOSS CALCULATION
        return sum((v**2).mean() for v in pred["mean"].values())

    def step(self):
        self.optimizer.zero_grad()
        plan = self.get_plan_for_pred()
        loss = self.get_loss(plan)
        loss.backward()
        self.optimizer.step()

        # update original plan dictionary in place
        for group, tensor in self.to_constrained(self.plan_param_uncons).items():
            self.plan[group].mul_(0.0).add_(tensor.detach())
