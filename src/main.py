import flet as ft
import pymupdf as pd
import os
from typing import List


def get_page_paths() -> List[str]:
    """
    Returns a list of image paths, each of which represents a page from the loaded PDF file. The PNG images live in ~/temp/.
    """
    paths = sorted([os.path.abspath(os.path.join("temp", fname)) for fname in os.listdir("temp")])
    # print(paths)
    return paths

def clear_temp_folder() -> None:
    """
    Deletes all images (pdf pages) from ~/temp/.
    """
    for image_path in get_page_paths():
        os.remove(image_path)

def pages_to_images(pdf: pd.Document, pdf_name: str) -> None:
    """
    Retrieves pages from the pdf file, converts them to PNGs and saves to ~/temp/.
    """
    for i, page in enumerate(pdf):
        page_png = page.get_pixmap(dpi=150)
        page_png.save(f"temp/{pdf_name[:9]}_{i:04d}.png")
    print("--File info retrieved!--")

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    file_column = ft.Column(controls=[],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            expand=True, scroll=ft.ScrollMode.AUTO)

    def on_dialog_result(e: ft.FilePickerResultEvent) -> None:
        # only process if user opened a file, not cancelled
        if e.files != None:
            # clear existing pdf
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
    submenu_file = ft.SubmenuButton(content=ft.Text(value="File", text_align=ft.TextAlign.CENTER), controls=submenu_file_controls)
    submenu_view = ft.SubmenuButton(content=ft.Text(value="View", text_align=ft.TextAlign.CENTER))
    submenu_chat = ft.SubmenuButton(content=ft.Text(value="Chat", text_align=ft.TextAlign.CENTER))

    menu_controls = [submenu_file, submenu_view, submenu_chat]
    menu = ft.MenuBar(menu_controls, expand=True)
    menubar = ft.Row([menu])

    

    

    # render everything
    page.add(menubar, file_column)

ft.app(main)
