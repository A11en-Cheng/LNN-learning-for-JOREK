"""Isolated fluid-head candidate; source-free inputs are not data certification.

Keep the existing full-native residual field implementation and CfC formula.
This Interface has no marker/shard/source arguments and does not run their
events. New data, forcing, physical boundary conditions and training remain
unqualified until separately audited. No conversion of coupled checkpoints.
"""
from copy import deepcopy

from torch import nn

from .native_field_step import NativeFieldStep


class NativePureFluidField(nn.Module):
    """Explicit field-only Adapter with a distinct scope/checkpoint fingerprint.

    Construct a fresh NativeFieldStep with compatible features and disabled
    SPI scenario encoding. Do not strip modules from a trained coupled model
    and call that an equivalent checkpoint migration.
    """
    def __init__(self, field):
        super().__init__()
        if type(field) is not NativeFieldStep:
            raise ValueError('pure-fluid candidate requires a standalone NativeFieldStep')
        self.field = field
        self._validate_contract()

    def _validate_contract(self):
        if self.field.temporal_core is None or self.field.feature_path is None:
            raise ValueError('explicit temporal core and compatible field-gradient path required')
        path = self.field.feature_path
        if path.scenario_enabled or path.scenario_encoder is not None or self.field.scenario_film is not None:
            raise ValueError('pure-fluid candidate rejects SPI scenario encoding')

    @property
    def units(self):
        return self.field.units

    @property
    def geometry(self):
        return self.field.geometry

    @property
    def encoder(self):
        return self.field.encoder

    @property
    def decoder(self):
        return self.field.decoder

    @property
    def temporal_core(self):
        return self.field.temporal_core

    @property
    def feature_path(self):
        return self.field.feature_path

    @property
    def state_scale(self):
        return self.field.state_scale

    @property
    def increment_scale(self):
        return self.field.increment_scale

    @property
    def center(self):
        return self.field.center

    @property
    def update_mapping(self):
        return self.field.update_mapping

    @property
    def checkpoint_configuration(self):
        self._validate_contract()
        config = deepcopy(self.field.checkpoint_configuration)
        config.update(pure_fluid_version='native-pure-fluid-v1',
            task_scope='pure_fluid_candidate',
            source_policy='not_accepted_not_zero_imputed',
            feedback_policy='not_accepted',
            source_events='none', scenario_parameters='disabled',
            data_qualification='separate_audit_required')
        return config

    def forward(self, values, raw_fluid_deltas, *, dt_si, previous_dt_si,
                scheme, time_coefficients=None, feature_parameters=None,
                memory=None, observation_values=None):
        self._validate_contract()
        if feature_parameters is not None:
            from .native_features import FeatureParameters
            if not isinstance(feature_parameters, FeatureParameters) or feature_parameters.scenario_enabled:
                raise ValueError('pure-fluid head requires explicitly routed non-SPI parameters')
        return self.field(values, raw_fluid_deltas, dt_si=dt_si,
            previous_dt_si=previous_dt_si, scheme=scheme,
            time_coefficients=time_coefficients, feature_parameters=feature_parameters,
            memory=memory, observation_values=observation_values)
