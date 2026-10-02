from ten_runtime import Addon,register_addon_as_extension
from openspline.integrations.ten import AvatarExtension
@register_addon_as_extension('openspline_avatar')
class OpensplineAddon(Addon):
    def on_create_instance(self,ten_env,name,context):
        ten_env.on_create_instance_done(AvatarExtension(name),context)
