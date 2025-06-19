import sys, os

# Add the project root to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

import flet as ft
from typing import Any
from src.backend.service import PDFService

LOGO_PATH = "/Users/illiakozlov/ChatWithPDF/chat-with-pdf/src/assets/logo.png"

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Initialize backend service, that also initializes the agent
    service = PDFService()

    file_column = ft.Column(controls=[ft.Image(src=LOGO_PATH, width=500, opacity=0.5)],
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

    def debug_parsed_content(content: Any) -> None:
        """
        This is a temporary function for debugging purposes.
        """
        chat_messages.controls.clear()
        chat_messages.controls.append(ft.Text(content))
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
        """
        Kicks off the PDF loading process when a file is selected.
        Renders pages as images in the UI.
        """
        if e.files != None:
            # clear old pdf from UI
            file_column.controls.clear()
            # load new pdf (returns a list of image paths to use in the UI)
            image_paths = service.load_pdf(e.files[0].path)
            image_pages = [ft.Image(src=path, fit=ft.ImageFit.CONTAIN) for path in image_paths]
            image_containers = [ft.Container(content=image_page, padding=10) for image_page in image_pages]
            file_column.controls.extend(image_containers)
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

    app_content = ft.Row([file_column, sidebar_handle, sidebar], spacing=0, expand=True)

    ui = ft.Column(controls=[menubar, app_content], spacing=0, expand=True)

    page.on_resized = on_window_resize

    # render everything
    page.add(ui)

    

ft.app(main)

