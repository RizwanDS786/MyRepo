#!/usr/bin/env python
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from oauth2client import file, client, tools
from httplib2 import Http
import gspread
import json


# Return List of Spreadsheets in a directory choosen by user
def get_spreadsheets(drive_sub_dir_name):
    directories = get_drive_directory_metadata()
    directory_id = get_directory_id_by_name(directories, drive_sub_dir_name)
    spreadsheet_list = get_spreadsheets_by_directory_id(directory_id)

    return spreadsheet_list

# Return list of directory metadata in drive
def get_drive_directory_metadata(SERVICE):
    payload = SERVICE.files().list(
        q=" mimeType = 'application/vnd.google-apps.folder' and \
                          trashed = false and \
                          %r in parents " % releases_dir_id,  # can add sharedWithMe = true
        fields='files(id, name)').execute()

    return payload.get('files', [])

# Return directory_id from directories matching drive_sub_dir_name
def get_directory_id_by_name(directories, drive_sub_dir_name):
    for directory in directories:
        if directory.get('name') == drive_sub_dir_name:
            directory_id = str(directory.get('id'))
            return directory_id

# Return list of spreadsheets in directory
def get_spreadsheets_by_directory_id(directory_id, SERVICE):
    payload = SERVICE.files().list(
        q=" mimeType = 'application/vnd.google-apps.spreadsheet' and \
                          trashed  = false and \
                          %r in parents " % directory_id,
        fields='files( id, name)').execute()

    return payload.get('files', [])


def main():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    store = file.Storage('E:\Project\storage.json')
    creds = store.get()

    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('C:/creds.json', scope)
        creds = client.OAuth2Credentials
        creds = tools.run_flow(flow, store)

    SERVICE = build('drive', 'v3', http=creds.authorize(Http()))
    releases_dir_id = '0B-31_xdQ4lnzY1B5eWRhVElOdHc'

    payload = SERVICE.files().list(
        q=" mimeType = 'application/vnd.google-apps.folder' and \
                            trashed = false and \
                            %r in parents " % releases_dir_id,  # can add sharedWithMe = true
        fields='files(id, name)').execute()

    directories = payload.get('files', [])
    test = map(lambda x: x['name'], directories)
    print(test)

    client_gspread = gspread.authorize(creds)

    drive_sub_dir_name = "GA-2.0.0"  # input("enter the sub directory name")
    directory_id = get_directory_id_by_name(directories, drive_sub_dir_name)
    spreadsheet_list = get_spreadsheets_by_directory_id(directory_id, SERVICE)
    print(spreadsheet_list)

    sheet_arr = list()
    summary_sheet = client_gspread.create('sheet_new').sheet1
    print("New sheet is Created")
    print(type(spreadsheet_list))
    summary_sheet.clear()
    bool = True
    # open all the sheets in the folder
    for i in spreadsheet_list:
        try:
            spreadsheet_name = i['name']
            sh = client_gspread.open(spreadsheet_name)
            sum_sheet = sh.worksheet('Summary')
            cell_val = sum_sheet.find("Total")
            if (bool == True):
                sp_name = sum_sheet
                bool = False
        except:
            continue
        print(cell_val)
        print(type(cell_val))
        print("Found something at R%sC%s" % (cell_val.row, cell_val.col))

        summary_sheet.insert_row(sum_sheet.row_values(cell_val.row))
        #print("values of sheet", sum_sheet.row_values(cell_val.row))
        #print(cell_val)
        #print(spreadsheet_name)
        cell_val_2 = summary_sheet.find("Total")
        summary_sheet.update_cell(cell_val_2.row, cell_val_2.col, spreadsheet_name)

    # To clear contents of the newly created sheet
    summary_sheet.insert_row(list())
    summary_sheet.insert_row(sp_name.row_values(1))
    summary_sheet.update_cell(1, 1, " ")

if __name__ == "__main__":
    main()