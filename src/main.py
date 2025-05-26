import flet as ft
import styles



def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0

    file_name = ft.Text("None")
    file_path = ft.Text("None")
    file_info = ft.Column([file_name, file_path])

    def on_dialog_result(e: ft.FilePickerResultEvent) -> tuple:
        if e.files != None:
            file_name.value = e.files[0].name
            file_path.value = e.files[0].path
            file_info.update()
            print("file info retrieved!")

    def open_file(e) -> None:
        file_picker.pick_files(initial_directory="Desktop", allowed_extensions=["pdf"])
        print("file dialog opened!")

    # file picker
    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)
    page.update()

    submenu_file_controls = [ft.MenuItemButton(content=ft.Text("Open"), close_on_click=True, on_click=open_file)]
    submenu_file = ft.SubmenuButton(content=ft.Text(value="File", text_align=ft.TextAlign.CENTER), controls=submenu_file_controls)
    submenu_view = ft.SubmenuButton(content=ft.Text(value="View", text_align=ft.TextAlign.CENTER))
    submenu_chat = ft.SubmenuButton(content=ft.Text(value="Chat", text_align=ft.TextAlign.CENTER))

    menu_controls = [submenu_file, submenu_view, submenu_chat]
    menu = ft.MenuBar(menu_controls, expand=True)
    menubar = ft.Row([menu])




    

    # render everything
    page.add(menubar, file_info)

ft.app(main)
