FOR /L %%i IN (0, 1, 5) DO (
	FOR /L %%j IN (0, 1, 3) DO (
		echo %%i %%j >> result_2.txt
		Project1.exe 4 6 6 2 5 %%i %%j < cycle.in.318 >> result_2.txt
	)
)