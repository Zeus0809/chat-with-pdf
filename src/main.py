import flet as ft
import pymupdf as pd
import os
from typing import List

# global document handle
pdf = None

def get_page_paths() -> List[str]:
    """
    Returns a list of image paths, each of which represents a page from the loaded PDF file. The PNG images live in ~/storage/temp/.
    """
    paths = sorted([os.path.abspath(os.path.join("storage/temp", fname)) for fname in os.listdir("storage/temp")])
    # print(paths)
    return paths

def clear_temp_folder() -> None:
    """
    Deletes all images (pdf pages) from ~/storage/temp/.
    """
    for image_path in get_page_paths():
        os.remove(image_path)

def pages_to_images(pdf: pd.Document, pdf_name: str) -> None:
    """
    Retrieves pages from the pdf file, converts them to PNGs and saves to ~/storage/temp/.
    """
    for i, page in enumerate(pdf):
        page_png = page.get_pixmap(dpi=150)
        page_png.save(f"storage/temp/{pdf_name[:9]}_{i:04d}.png")
    print("--New file info retrieved!--")

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    file_column = ft.Column(controls=[],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            expand=True, scroll=ft.ScrollMode.AUTO)
    
    sidebar = ft.Container(
        content=ft.Text("Sidebar Placeholder", text_align=ft.TextAlign.CENTER),
        width=0,
        height=page.window.height,
        bgcolor=ft.Colors.BLUE_50,
        border_radius=ft.border_radius.all(5),
        padding=10,
        animate=ft.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
    )

    def toggle_sidebar(e):
        """
        Toggles the sidebar visibility.
        """
        if sidebar.width == 0:
            sidebar.width = 350 # expand
        else:
            sidebar.width = 0 # collapse
        sidebar.update()

    def on_dialog_result(e: ft.FilePickerResultEvent) -> None:
        # only process if user opened a file, not cancelled
        if e.files != None:
            # clear existing pdf
            global pdf
            if pdf is not None:
                pdf.close()
                print("--Old file closed!--")
                file_column.controls.clear()
                clear_temp_folder()
            # open new pdf, parse into PNG images, add to UI to render
            pdf = pd.open(e.files[0].path)
            pages_to_images(pdf, e.files[0].name)
            paths = get_page_paths()
            image_pages = [ft.Image(src=path, fit=ft.ImageFit.CONTAIN) for path in paths]
            file_column.controls.extend(image_pages)
            file_column.update()
            print(f"--{len(file_column.controls)} pages from {e.files[0].name} rendered!--")

    def open_file(e) -> None:
        file_picker.pick_files(initial_directory="Desktop", allowed_extensions=["pdf"])
        print("--File dialog opened!--")

    # overlay
    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)
    page.update()

    submenu_file_controls = [ft.MenuItemButton(content=ft.Text("Open"), close_on_click=True, on_click=open_file)]
    submenu_chat_controls = [ft.MenuItemButton(content=ft.Text("Toggle Chat"), close_on_click=False, on_click=toggle_sidebar)] 
    submenu_file = ft.SubmenuButton(content=ft.Text(value="File", text_align=ft.TextAlign.CENTER), controls=submenu_file_controls)
    submenu_view = ft.SubmenuButton(content=ft.Text(value="View", text_align=ft.TextAlign.CENTER))
    submenu_chat = ft.SubmenuButton(content=ft.Text(value="Chat", text_align=ft.TextAlign.CENTER), controls=submenu_chat_controls)

    menu_controls = [submenu_file, submenu_view, submenu_chat]
    menu = ft.MenuBar(menu_controls, expand=True)
    menubar = ft.Row([menu])

    app_content = ft.Row([file_column, sidebar], expand=True)

    

    # render everything
    page.add(menubar, app_content)

ft.app(main)
