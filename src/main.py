import flet as ft
import styles

def main(page: ft.Page):
    page.title = "Chat With PDF"
    page.padding = 0
    #page.spacing = 0
    #page.vertical_alignment = ft.MainAxisAlignment.CENTER

    submenu_file = ft.SubmenuButton(content=ft.Text(value="File", text_align=ft.TextAlign.CENTER))
    submenu_view = ft.SubmenuButton(content=ft.Text(value="View", text_align=ft.TextAlign.CENTER))
    submenu_chat = ft.SubmenuButton(content=ft.Text(value="Chat", text_align=ft.TextAlign.CENTER))

    menu_controls = [submenu_file, submenu_view, submenu_chat]
    menu = ft.MenuBar(menu_controls, expand=True)
    menubar = ft.Row([menu])
    








    # render everything
    page.add(menubar)

ft.app(main)
