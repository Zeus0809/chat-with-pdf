import sys, os, time, threading

# Add the project root to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

import flet as ft
from dotenv import load_dotenv
from src.backend.service import PDFService
from src.backend.agent import PDFAgent
from styles import ChatStyles, TextStyles, Dimensions

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    load_dotenv()

    # Initialize backend service, that also initializes the agent (using docker model runner by default)
    service = PDFService()

    def on_message_send(e) -> None:
        """
        Handles submitting user prompt to the agent and receiving the response.
        """
        if message_input.value.strip() == "":
            return

        user_message = message_input.value.strip()
        user_text_block = ft.Text(f"{user_message}", **TextStyles.message_text())
        user_row = ChatStyles.create_user_message_row(bubble_content=user_text_block)
        chat_messages.controls.append(user_row)
        message_input.value = ""
        # Show loading indicator
        loading_agent()
        # Disable send button during processing
        send_button.disabled = True
        sidebar_content.update()
        chat_messages.scroll_to(offset=-1, curve=ft.AnimationCurve.EASE_OUT)
        
        # Ask the agent
        start_time = time.time()
        response = service.agent.ask_agent(user_message) # response is a generator object
        
        # Create placeholder to accumulate response, and a flag to wait for first token arrival
        agent_text_block = ft.Text("", **TextStyles.message_text())
        agent_row = ChatStyles.create_agent_message_row(bubble_content=agent_text_block)
        first_token = True
        for text in response.response_gen:
            if first_token:
                del chat_messages.controls[-1] # remove loading
                chat_messages.controls.append(agent_row)
                chat_messages.update()
                first_token = False
            # display the rest of the stream
            if "\n" in text:
                text = text.strip("\n")
            agent_row.controls[0].controls[0].content.value += text
            agent_row.update()
            chat_messages.scroll_to(offset=-1, curve=ft.AnimationCurve.EASE_OUT)
        # add elapsed time
        elapsed_time = time.time() - start_time
        elapsed_time_text = ft.Text(f"({elapsed_time:.2f}s)", **TextStyles.elapsed_time())
        agent_row.controls[0].controls.append(elapsed_time_text) # add time block to the chat bubble column after bubble container, so it appears below agent text        

        # Re-enable send button
        send_button.disabled = False
        sidebar_content.update()
        chat_messages.scroll_to(offset=-1, curve=ft.AnimationCurve.EASE_OUT)

    def loading_agent() -> None:
        """
        Create and add a loading indicator for agent responses to the UI.
        """
        loading_container = ft.Row(
            controls=[
                ft.ProgressRing(width=20, height=20, stroke_width=2),
                ft.Text("Agent is thinking...", **TextStyles.loading_text())
            ],
            spacing=10,
            alignment=ft.MainAxisAlignment.START
        )
        chat_messages.controls.append(loading_container)

    def loading_file() -> ft.ProgressRing:
        """
        Create and add to the UI a loading indicator when waiting for the pdf to load (including index creation).
        """
        ring=ft.ProgressRing(
            width=100,
            height=100,
            stroke_width=5,
            badge=ft.Badge(
                text="Getting your PDF ready...",
                alignment=ft.Alignment(x=-1.7, y=2),
                bgcolor=ft.Colors.TRANSPARENT,
                text_color="#316093"
            ),
            expand=True
        )
        loading_container = ft.Container(
            content=ring,
            alignment=ft.Alignment(x=0, y=0)
        )
        file_column.controls.append(loading_container)
        file_column.update()
        return ring

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
            sidebar.width = Dimensions.SIDEBAR["start_width"] # expand
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
            # add loading indicator
            progress_ring = loading_file()

            stop_animation = [False] # flag to stop animation early if pdf loading overtook it

            def animate_progress():
                progress = 0
                while not stop_animation[0] and progress < 0.95:
                    progress += 0.0095
                    progress_ring.value = progress
                    file_column.update()
                    time.sleep(0.1)
                # Hold at 95% to wait for pdf to finish loading
                pulse_values = [0.93, 0.95, 0.97, 0.95]
                pulse_index = 0
                while not stop_animation[0]:
                    progress_ring.badge.text = "It's a large file, hang tight..."
                    progress_ring.value = pulse_values[pulse_index]
                    file_column.update()
                    pulse_index = (pulse_index + 1) % len(pulse_values) # this cycles through values 0-1-2-3-0-...
                    time.sleep(0.3)

            # start animation in a new thread
            threading.Thread(target=animate_progress, daemon=True).start()

            # load new pdf (returns a list of image paths to use in the UI)
            image_paths = service.load_pdf(e.files[0].path)

            # complete animation
            stop_animation[0] = True
            progress_ring.value = 1.0
            file_column.update()
            time.sleep(0.1)

            image_pages = [ft.Image(src=path, fit=ft.ImageFit.CONTAIN) for path in image_paths]
            image_containers = [ft.Container(content=image_page, padding=10) for image_page in image_pages]
            file_column.controls.clear() # remove loading ring
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

    file_column = ft.Column(
        controls=[
            ft.Image(src=os.getenv("LOGO_PATH"), width=500, opacity=0.5)
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True,
        scroll=ft.ScrollMode.AUTO
    )

    chat_messages = ft.Column(
        controls=[],
        expand=True,
        scroll=ft.ScrollMode.ALWAYS,
    )

    message_input = ft.TextField(
        hint_text="Ask a question here...",
        multiline=True,
        expand=True,
        shift_enter=True,
        on_submit=on_message_send
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

    sidebar_content = ft.Column(
        controls=[
            ft.Container(content=chat_messages, expand=5),
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
        content=sidebar_content,
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
    # submenu_view = ft.SubmenuButton(content=ft.Text(value="View", text_align=ft.TextAlign.CENTER))
    submenu_chat = ft.SubmenuButton(content=ft.Text(value="Chat", text_align=ft.TextAlign.CENTER), controls=submenu_chat_controls)

    menu_controls = [submenu_file, submenu_chat]
    menu = ft.MenuBar(menu_controls, expand=True)
    menubar = ft.Row([menu])

    app_content = ft.Row([file_column, sidebar_handle, sidebar], spacing=0, expand=True)

    ui = ft.Column(controls=[menubar, app_content], spacing=0, expand=True)

    page.on_resized = on_window_resize

    # render everything
    page.add(ui)

ft.app(main)

