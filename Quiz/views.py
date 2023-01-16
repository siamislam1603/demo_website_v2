from django.shortcuts import render, redirect
from .models import *
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .forms import QuizForm, QuestionForm
from django.forms import inlineformset_factory
from django.views.generic import UpdateView
import random


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


def index(request):
    quiz = Quiz.objects.all()
    print("quiz",quiz)
    para = {'quiz': quiz}
    return render(request, "index_page.html", para)


@login_required(login_url='/login')
def quiz(request, slug):
    quiz = Quiz.objects.get(slug=slug)
    print(quiz.name)
    return render(request, "quiz.html", {'quiz': quiz})


def quiz_data_view(request, slug):
    quiz = Quiz.objects.get(slug=slug)

    questions = []
    images = []

    for q in quiz.get_questions():
        answers = []
        images.append(q.image.url)
        for a in q.get_answers():
            # print(a)
            answers.append(a.content)
        try:
            random.shuffle(answers)
        except:
            pass
        questions.append({str(q): answers})
    #     choosen =[]
    #     already=[]
    #     g = 0
    # print(len(questions))
    # for i in range(0,7):
    #     # g = random.randint(0, 6)
    #     while g in already:
    #         g = random.randint(0, 6)
    #     choosen.append(questions[g])
    #     already.append(g)

        # print(questions)
    questions=random.sample(questions, 7)
    temp = list(zip(questions, images))
    random.shuffle(temp)
    questions, images = zip(*temp)
    print(images)

    return JsonResponse({
        'data': questions[:7],
        'time': quiz.time,
        'images':images[:7],
    })


def save_quiz_view(request, slug):
    if is_ajax(request):
        questions = []
        data = request.POST
        data_ = dict(data.lists())

        data_.pop('csrfmiddlewaretoken')

        for k in data_.keys():
            print('key: ', k)
            question = Question.objects.get(content=k)
            questions.append(question)

        user = request.user
        quiz = Quiz.objects.get(slug=slug)

        score = 0
        marks = []
        correct_answer = None

        for q in questions:
            a_selected = request.POST.get(q.content)

            if a_selected != "":
                question_answers = Answer.objects.filter(question=q)
                for a in question_answers:
                    if a_selected == a.content:
                        if a.correct:
                            score += 1
                            correct_answer = a.content
                    else:
                        if a.correct:
                            correct_answer = a.content

                marks.append({str(q): {'correct_answer': correct_answer, 'answered': a_selected}})
            else:
                marks.append({str(q): 'not answered'})

        Marks_Of_User.objects.create(quiz=quiz, user=user, score=score)

        return JsonResponse({'passed': True, 'score': score, 'marks': marks})


def Signup(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        password = request.POST['password1']
        confirm_password = request.POST['password2']

        if password != confirm_password:
            return redirect('/register')

        user = User.objects.create_user(username, email, password)
        user.first_name = first_name
        user.last_name = last_name
        user.save()
        return render(request, 'login.html')
    return render(request, "signup.html")


def Login(request):
    if request.user.is_authenticated:
        return redirect('/')
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            return render(request, "login.html")
    return render(request, "login.html")


def Logout(request):
    logout(request)
    return redirect('/')


def add_quiz(request):
    quizs = Quiz.objects.all()
    quizs = Quiz.objects.filter().order_by('-id')
    if request.method == "POST":
        form = QuizForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, "add_quiz.html")
    else:
        form = QuizForm()
    return render(request, "add_quiz.html", {'form': form, 'quizs': quizs})

def delete_quiz(request, myid):
    quiz = Quiz.objects.get(id=myid)
    if request.method == "POST":
        quiz.delete()
        return redirect('/add_quiz')
    return render(request, "delete_quiz.html", {'question': quiz})

# def add_quiz(request):
#     if request.method == "POST":
#         form = QuizForm(data=request.POST)
#         if form.is_valid():
#             quiz = form.save(commit=False)
#             quiz.save()
#             obj = form.instance
#             return render(request, "add_quiz.html", {'obj': obj})
#     else:
#         form = QuizForm()
#     return render(request, "add_quiz.html", {'form': form})


def add_question(request):
    questions = Question.objects.all()
    questions = Question.objects.filter().order_by('-id')
    if request.method == "POST":
        form = QuestionForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, "add_question.html")
    else:
        form = QuestionForm()
    return render(request, "add_question.html", {'form': form, 'questions': questions})


def delete_question(request, myid):
    question = Question.objects.get(id=myid)
    if request.method == "POST":
        question.delete()
        return redirect('/add_question')
    return render(request, "delete_question.html", {'question': question})


class QuizUpdatePostView(UpdateView):
    model = Quiz
    template_name = 'edit_quiz.html'
    fields = ['name', 'slug', 'desc', 'number_of_questions','time']

class UpdatePostView(UpdateView):
    model = Question
    template_name = 'add_options.html'
    fields = ['content', 'slug', 'image', 'quiz']

def add_options(request, myid):
    question = Question.objects.get(id=myid)
    QuestionFormSet = inlineformset_factory(Question, Answer, fields=('content', 'correct', 'question'), extra=4)
    if request.method == "POST":
        formset = QuestionFormSet(request.POST, instance=question)
        if formset.is_valid():
            formset.save()
            alert = True
            return render(request, "add_ans.html", {'alert': alert})
    else:
        formset = QuestionFormSet(instance=question)
    return render(request, "add_ans.html", {'formset': formset, 'question': question})


def LeaderBoard(request):
    marks = Marks_Of_User.objects.all()
    marks = Marks_Of_User.objects.filter().order_by('-score')
    return render(request, "result_user.html", {'marks': marks})

def results(request):
    marks = Marks_Of_User.objects.all()
    marks = Marks_Of_User.objects.filter().order_by('-score')
    return render(request, "results.html", {'marks': marks})


def delete_result(request, myid):
    marks = Marks_Of_User.objects.get(id=myid)

    if request.method == "POST":
        marks.delete()
        return redirect('/results')
    return render(request, "delete_result.html", {'marks': marks})
