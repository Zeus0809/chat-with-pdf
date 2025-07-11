import sys, os, time

# Add the project root to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

import flet as ft
from src.backend.service import PDFService

LOGO_PATH = "/Users/illiakozlov/ChatWithPDF/chat-with-pdf/src/assets/logo.png"

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    # Initialize backend service, that also initializes the agent and parser
    service = PDFService()

    def on_message_send(e) -> None:
        """
        Handles submitting user prompt to the agent and receiving the response.
        """
        if message_input.value.strip() == "":
            return
        else:
            # chat_content.controls[0].border = ft.border.all(1, ft.Colors.BLACK)
            user_message = message_input.value.strip()
            chat_messages.controls.append(ft.Text(f"You: {user_message}", text_align=ft.TextAlign.RIGHT))
            message_input.value = ""
            # Show loading indicator
            loading_container = ft.Container(
                content=ft.Row([
                    ft.ProgressRing(width=20, height=20, stroke_width=2),
                    ft.Text("Agent is thinking...", size=14, color=ft.Colors.GREY_600)
                ], spacing=10, alignment=ft.MainAxisAlignment.START),
                padding=ft.padding.only(left=10, top=5, bottom=5)
            )
            chat_messages.controls.append(loading_container)
            # Disable send button during processing
            send_button.disabled = True
            chat_content.update()
            
            # Ask the agent
            start_time = time.time()
            response = service.agent.ask_agent(user_message)
            elapsed_time = time.time() - start_time
            
            # Remove loading indicator and add response
            chat_messages.controls.pop()  # Remove loading indicator
            chat_messages.controls.append(ft.Text(f"Agent ({elapsed_time:.2f}s): {response}", text_align=ft.TextAlign.JUSTIFY))
            # Re-enable send button
            send_button.disabled = False
            chat_content.update()

    def display_parsed_content(content: str) -> None:
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
            # debug parsed content
            # display_parsed_content(service.parser.debug_chunks())
            # agent chat test

    def open_file(e) -> None:
        file_picker.pick_files(initial_directory="Desktop", allowed_extensions=["pdf"])
        print("--File dialog opened!--")

#############-UI-Elelments-###############

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
        on_click=on_message_send
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

