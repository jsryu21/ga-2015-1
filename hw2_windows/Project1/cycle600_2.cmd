FOR /L %%i IN (8, 1, 11) DO (
	start Project1.exe 1 3 8 0 2 4 2 < cycle.in.600 >> out_600_%%i.txt
)