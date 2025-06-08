import flet as ft
import pymupdf as pd
import os
from typing import List
import time
import parser

# global raw document handle
pdf = None

# global structured document handle
document_content = None

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

def parse_document_into_blocks() -> None:
        """
        Parses the loaded PDF into structured content blocks using parser.build_page_content(). 
        """
        assert pdf is not None, "No PDF document loaded: global handle is None."
        print("\n--Parsing document...--\n")
        global document_content
        document_content = []
        total_t = 0
        for page in pdf:
            start = time.time()
            page_content = parser.build_page_content(page)
            document_content.append(page_content)
            end = time.time()
            t = round(end-start, 2)
            total_t += t
            print(f"--Page {page.number+1} parsed in {t}s.--")
        print(f"\n--Document parsed in {round(total_t, 2)}s.--\n")

def parse_document_into_markdown() -> None:
        """
        Parses the loaded PDF into markdown using parser.get_page_markdown(). 
        """
        assert pdf is not None, "No PDF document loaded: global handle is None."
        print("\n--Parsing document...--\n")
        global document_content
        starttime = time.time()
        document_content = parser.get_doc_markdown(pdf)
        endtime = time.time()
        print(f"\n--Document parsed in {round(endtime-starttime, 2)}s.--\n")

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    file_column = ft.Column(controls=[],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                            expand=True, scroll=ft.ScrollMode.AUTO)

    chat_messages = ft.Column(
        controls=[],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )

    message_input = ft.TextField(
        hint_text="Ask a question here...",
        multiline=True,
        expand=True,
    )

    send_button = ft.IconButton(
        icon=ft.Icons.SEND,
        tooltip="Send Message",
    )

    input_row = ft.Row([message_input, send_button],
                       spacing=10,
                       vertical_alignment=ft.CrossAxisAlignment.END,
                       expand=True)

    chat_content = ft.Column(
        controls=[
            ft.Container(content=chat_messages, expand=3),
            ft.Container(content=input_row, expand=1)
        ],
        spacing=10,
        expand=True
    )

    sidebar_handle = ft.GestureDetector(
        content=ft.Container(
            width=5,
            bgcolor=ft.Colors.BLUE_300
        ),
        drag_interval=1,
        on_pan_update=lambda e: resize_sidebar(e),
        mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT
    )

    sidebar = ft.Container(
        content=chat_content,
        width=0,
        bgcolor=ft.Colors.BLUE_50,
        padding=10,
    )

    def debug_parsed_content() -> None:
        """
        This is a temporary function for debugging purposes.
        Builds a string representation of the parsed content and puts it into the sidebar for debugging.
        """
        chat_messages.controls.clear()
        content_string = ""
        for page in document_content:
            for block in page:
                content_string += str(block) + "\n"
        chat_messages.controls.append(ft.Text(content_string))
        chat_messages.update()

    def on_window_resize(e: ft.WindowResizeEvent) -> None:
        """
        Handles window resize events to adjust sidebar height.
        """
        sidebar.height = page.window.height
        sidebar_handle.height = page.window.height
        sidebar.update()
        sidebar_handle.update()

    def resize_sidebar(e: ft.DragUpdateEvent) -> None:
        """
        Implements drag resize functionality for the sidebar
        """
        sidebar.animate = None
        new_width = sidebar.width - e.delta_x
        if 0 <= new_width <= page.window.width * 0.5: # limit width to 50% of window
            sidebar.width = new_width
            sidebar.update()

    def toggle_sidebar(e) -> None:
        """
        Toggles the sidebar visibility.
        """
        if sidebar.animate is None:
            sidebar.animate = ft.Animation(200, ft.AnimationCurve.EASE_IN_OUT)
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
            begin = time.time()
            pdf = pd.open(e.files[0].path)
            pages_to_images(pdf, e.files[0].name)
            paths = get_page_paths()
            image_pages = [ft.Image(src=path, fit=ft.ImageFit.CONTAIN) for path in paths]
            image_containers = [ft.Container(content=image_page, padding=10) for image_page in image_pages]
            file_column.controls.extend(image_containers)
            file_column.update()
            end = time.time()
            print(f"--{len(file_column.controls)} pages from {e.files[0].name} rendered in {round(end-begin, 2)}s!--")
            # parse the document into markdown format
            parse_document_into_markdown()
            # debugging
            chat_messages.controls.clear()
            chat_messages.controls.append(ft.Text(document_content))
            chat_messages.update()

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

    app_content = ft.Row([file_column, sidebar_handle, sidebar], spacing=0, expand=True)

    ui = ft.Column(controls=[menubar, app_content], spacing=0, expand=True)

    page.on_resized = on_window_resize

    # render everything
    page.add(ui)


ft.app(main)
