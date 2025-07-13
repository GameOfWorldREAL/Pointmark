from src.utils.JsonManager import JsonManager

def print_settings_found(settings_json: JsonManager):
    print("settings found:")
    print("---------------------------------------")
    print(settings_json)
    print("---------------------------------------")

#check if a project is a batch project
def is_batch(settings_json: JsonManager) -> bool:
    batch = settings_json.get_value("batch")
    if batch is None:
        raise Exception("settings seems corrupted")
    if batch:
        return True
    else:
        return False

#return an empty settings shell
def build_empty_settings() -> dict:
    settings = {
        "batch": None,
        "project_path": None,
        "video_path": None,
        "keyframer": None,
        "reconst_pipeline": None,
        "meshroom_path": None,
        "template_path": None,
        "colmap_path": None
    }
    return settings

#return an empty batch_settings shell
def build_empty_batch_settings():
    settings = {
        "batch": None,
        "setup_path": None,
        "project_path": None,
        "video_path": None,
        "keyframer": None,
        "reconst_pipeline": None,
        "meshroom_path": None,
        "template_path": None,
        "colmap_path": None,
        "setup": None
    }
    return settings