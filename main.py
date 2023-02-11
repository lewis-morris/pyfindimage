from locater import Locator

lctr = Locator("move_bulk", "/home/lewis/fileStore", "/home/lewis/fileOut", "~/Downloads/prostock.xlsx")
lctr.load_files()
lctr.match_files()
df = lctr.reload_complete_df()
lctr.save_job()
lctr.check_matches()
lctr.check_move_partials()
lctr.check_move_partials()
