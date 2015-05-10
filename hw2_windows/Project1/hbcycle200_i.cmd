FOR /L %%i IN (0, 1, 3) DO (
	Project1.exe 5 11 11 8 10 4 2 < cycle.in.200 >> hb_out_200_%1.txt
)