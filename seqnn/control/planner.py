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


class CategoricalCEMPlanner:
    def __init__(
        self,
        model,
        data_past,
        control_plan,
        num_categories,
        population_size=128,
        num_elite=16,
        step_size=0.5,
    ):
        self.model = model
        self.past = {
            key: tensor.repeat(population_size, 1, 1)
            for key, tensor in data_past.items()
        }
        self.plan = control_plan
        self.plans = {
            key: tensor.repeat(population_size, 1, 1)
            for key, tensor in control_plan.items()
        }
        self.tags_to_optimize = list(num_categories.keys())
        plan_length = model.get_tags(control_plan, self.tags_to_optimize[0]).shape[1]
        self.probs = {
            tag: torch.ones(1, plan_length, 1, num_categ) / num_categ
            for tag, num_categ in num_categories.items()
        }
        self.num_categories = num_categories
        self.population_size = population_size
        self.num_elite = num_elite
        self.step_size = step_size

    def sample_plan_population(self):
        # sample each tag separately
        for tag in self.tags_to_optimize:
            probs = self.probs[tag].repeat(self.population_size, 1, 1, 1)
            sample = torch.distributions.Categorical(probs=probs).sample()
            self.model.set_tags(self.plans, tag, sample)
        return self.plans

    def get_losses(self, plan):
        pred = self.model.predict(self.past, plan)
        # TODO: IMPLEMENT HERE THE LOSS CALCULATION
        return sum((v**2).mean(dim=(1, 2)) for v in pred["mean"].values())

    def update_probs(self, ind_elite):
        for tag in self.tags_to_optimize:
            plans_elite_tag = self.model.get_tags(self.plans, tag)[ind_elite, :, :]
            for k in range(self.num_categories[tag]):
                estimate_new = (1.0 * (plans_elite_tag == k)).mean(dim=0, keepdim=True)
                estimate_old = self.probs[tag][:, :, :, k]
                self.probs[tag][:, :, :, k] = (
                    self.step_size * estimate_new + (1 - self.step_size) * estimate_old
                )

    def update_plan(self, ind_elite):
        ind_best = ind_elite[0]
        for tag in self.tags_to_optimize:
            plans_tag = self.model.get_tags(self.plans, tag)
            self.model.set_tags(
                self.plan, tag, plans_tag[ind_best : ind_best + 1, :, :]
            )

    def step(self):
        self.sample_plan_population()
        losses = self.get_losses(self.plans)
        ind_elite = (-losses).topk(self.num_elite).indices
        self.update_probs(ind_elite)
        self.update_plan(ind_elite)
