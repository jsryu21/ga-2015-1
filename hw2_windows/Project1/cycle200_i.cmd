FOR /L %%i IN (0, 1, 3) DO (
	Project1.exe 1 3 10 0 2 4 2 < cycle.in.200 >> out_200_%1.txt
)