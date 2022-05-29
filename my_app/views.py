from django.shortcuts import render

from my_app.song_recommender import get_music_data_test
def my_view(request):
    if request.method == 'GET':
        return render(request, 'my_app/home.html', {"data": None})
    else:
        user_input=request.POST.get("music_input")
        test_response= get_music_data_test(user_input)
        data = test_response.reset_index().to_dict(orient='records')
        return render(request, 'my_app/home.html', {"data": data, "user_input": user_input})
