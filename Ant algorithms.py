import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import math
import time

def probability(i,j,a,b,M,F,pos):
    return ( (1/(M[i,j]))**b * (F[i,j])**a )/sum((1/(M[i,k]))**b * (F[i,k])**a for k in pos)

def edges_in_path(path): # построить список ребер по вершинам в пути
    return [(path[i],path[i+1]) for i in range(len(path)-1)]

def choose_vertex(cur,a,b,M,F,pos): # выбрать следующую вершину
    probs=list(map(lambda x: probability(cur,x,a,b,M,F,pos), pos)) # список вероятностей выбрать определенную вершину
    return rand.choice(list(pos),None,p=probs)

def nearest_neighbour_tour(M,start):
    new_M = M.copy()
    cur=start
    l=0
    path=[cur]
    while len(path)!=len(M):
        new_M[:,cur] = 100
        new_cur = np.where(new_M[cur]==min(new_M[cur]))[0][0]
        l+=new_M[cur,new_cur]
        path.append(new_cur)
        cur=new_cur
    path.append(path[0]) # замыкаем путь муравья
    l+=M[cur,path[0]]
    return (l,path)

def ant_system_path(cur,a,b,q,M,F,pos): # построить путь для одного муравья
    l=0 # начальная длина пути
    path=[cur] # пройденный путь
    new_F=F.copy() # задаем изменяемую таблицу феромонов
    while pos!=set():
        new_cur=choose_vertex(cur,a,b,M,F,pos) # выбираем новую текущую вершину
        l+=M[cur,new_cur] # увеличиваем длину пути
        cur=new_cur
        pos.remove(cur)
        path.append(cur) # добавляем текущую вершину в путь
    path.append(path[0]) # замыкаем путь муравья
    l+=M[cur,path[0]]
    for edge in edges_in_path(path): # цикл по ребрам в пути
        new_F[edge] = new_F[edge[::-1]] = new_F[edge] +q/l # добавляем феромон по правилу
    return (path, l, new_F)

def ant_system(M,a=1,b=1,p=0.5,q=1,e=5):
    # a - степень при феромоне
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона
    # e - количество элитных муравьев

    count=0
    neighbour_tour = nearest_neighbour_tour(M,0)
    init_pos = set(range(len(M)))
    best_l=neighbour_tour[0]
    best_path=neighbour_tour[1]
    l_list=[]
    best_l_list=[best_l]
    cur_F=np.zeros(np.shape(M), dtype = np.float)
    cur_F.fill(len(M)/(nearest_neighbour_tour(M,0)[0]))
    cur_F =cur_F - np.diag(np.diag(cur_F))
    new_F=cur_F.copy()
    c=0
    while count<=100: # внешний цикл по кол-ву итераций
        for ant in range(len(M)): # цикл по муравьям(в каждом городе по муравью)
            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l, new_F = ant_system_path(cur,a,b,q,M,cur_F,pos) # строим путь, считаем длину и получаем новую матрицу феромона

            l_list.append(l)

            if l<=best_l:
                best_l=l
                best_path=path
        best_l_list.append(best_l)

        for edge in edges_in_path(best_path):
            new_F[edge] = new_F[edge[::-1]] = new_F[edge] +(q*e)/best_l

        #print('b',cur_F)
        add = new_F-cur_F
        cur_F=(1-p)*cur_F # происходит испарение феромона по всем ребрам
        cur_F += add
        #print('a',cur_F)
        count+=1
        if best_l==best_l_list[-2]:
            c+=1
            if c==20:
                print('Выход по времени')
                break
        else:
            c=0
        print(count,': ',best_l)
    #print(l_list)
    #print(cur_F)
    return (best_l, best_path,best_l_list)


def generate_symmetric(mat):
    m=mat.copy()
    for i in range(len(m)):
        for j in range(len(m)):
            m[j,i]=m[i,j]
    return m

def generate_points(x,y,n=10):
    return rand.uniform(x,y,size=(n,2))

def dist(two_points):
    a=two_points[0]
    b=two_points[1]
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def generate_weights(points):
    M=np.zeros((len(points),len(points)))
    for i in range(len(points)):
        for j in range(i,len(points)):
            M[i,j]=dist((points[i],points[j]))
    return generate_symmetric(M)

def points_in_path(points,path):
    return np.array(list(map(lambda x: points[x],path)))

def ant_colony_drawing(points,a=1,b=1,p=0.5,q=1,e=5):
    # a - степень при феромоне
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона
    # e - количество элитных муравьев
    M = generate_weights(points)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)

    count=0
    init_pos = set(range(len(M)))
    best_l=float('inf')
    best_path=[]
    #l_list=[]
    #best_l_list=[]
    cur_F=np.ones(np.shape(M), dtype = np.float)-np.eye(np.shape(M)[0], dtype = np.float)
    new_F=cur_F.copy()

    plt.ion()
    minx=min([i[0]for i in points])
    miny=min([i[1]for i in points])
    maxx=max([i[0]for i in points])
    maxy=max([i[1]for i in points])
    plt.axes([minx,miny,maxx,maxy])
    while count<=30: # внешний цикл по кол-ву итераций
        for ant in range(len(M)): # цикл по муравьям(в каждом городе по муравью)
            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l, new_F = ant_path(cur,a,b,q,M,cur_F,pos) # строим путь, считаем длину и получаем новую матрицу феромона

            #l_list.append(l)

            if l<=best_l:
                best_l=l
                best_path=path

            b_pts = points_in_path(best_path).transpose()
            pts = points_in_path(path).transpose()
            ax1.cla()
            ax1.plot(points.transpose()[0],points.transpose()[1],'ro',b_pts[0],b_pts[1],'b',linewidth = 3.0)
            ax1.plot(pts[0],pts[1],'g')
            ax1.text(2, 5, str(count)+ '  '+ str(best_l))
            fig.canvas.draw()
            plt.pause(0.01)
            plt.show()
                #best_l_list.append(best_l)

        for edge in edges_in_path(best_path):
           new_F[edge] = new_F[edge[::-1]] = new_F[edge] +(q*e)/l

        new_F = (1-p)*new_F # происходит испарение феромона по всем ребрам
        cur_F=new_F
        count+=1
    #print(l_list)
    print(cur_F)
    print(l,best_l)
    plt.close()
    return (best_l, best_path)

def get_best_vertex(cur,a,b,M,F,pos):
    probs=list(map(lambda x: [x,probability(cur,x,a,b,M,F,pos)], pos)) # список вероятностей выбрать определенную вершину
    return max(probs, key = lambda x: x[1])[0]

def acs_choose_vertex(cur,a,b,M,F,pos,method_coef): # выбрать следующую вершину
    method = rand.choice([0,1],None,p=[method_coef,1-method_coef])
    if method == 0:
        return get_best_vertex(cur,1,b,M,F,pos)
    else:
        probs=list(map(lambda x: probability(cur,x,1,b,M,F,pos), pos)) # список вероятностей выбрать определенную вершину
        return rand.choice(list(pos),None,p=probs)

def ant_colony_system_path(cur,a,b,q,M,F,pos,method_coef,ph_coef): # построить путь для одного муравья
    l=0 # начальная длина пути
    path=[cur] # пройденный путь
    new_F=F.copy() # задаем изменяемую таблицу феромонов
    while pos!=set():
        new_cur=acs_choose_vertex(cur,a,b,M,F,pos,method_coef) # выбираем новую текущую вершину
        l+=M[cur,new_cur] # увеличиваем длину пути
        new_F[cur,new_cur] = (1-ph_coef)*new_F[cur,new_cur] + q*ph_coef # локальное обновление феромона
        cur=new_cur
        pos.remove(cur)
        path.append(cur) # добавляем текущую вершину в путь

    path.append(path[0]) # замыкаем путь муравья
    new_F[cur,path[0]] = (1-ph_coef)*new_F[cur,path[0]] + q*ph_coef
    l+=M[cur,path[0]]
    return (path, l, new_F)

def ant_colony_system(M,method_coef,ph_coef,b=1,p=0.1,q=1):
    # method_coef - коэффициент выбора метода выбора вершины
    # ph_coef - коэффициент локального обновления феромона
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона

    count=0
    neighbour_tour = nearest_neighbour_tour(M,0)
    init_pos = set(range(len(M)))
    best_l=neighbour_tour[0]
    best_path=neighbour_tour[1]
    #l_list=[]
    best_l_list=[best_l]
    cur_F=np.zeros(np.shape(M), dtype = np.float)
    cur_F.fill(len(M)/(nearest_neighbour_tour(M,0)[0]))
    cur_F =cur_F - np.diag(np.diag(cur_F))
    new_F=cur_F.copy()
    c=0
    while count<=100: # внешний цикл по кол-ву итераций
        for ant in range(50): # цикл по муравьям(в каждом городе по муравью)
            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l, new_F = ant_colony_system_path(cur,1,b,q,M,cur_F,pos,method_coef,ph_coef) # строим путь, считаем длину и получаем новую матрицу феромона
            cur_F=new_F
            #l_list.append(round(l,2))
            #print(l,best_l)
            if l<=best_l:
                best_l=l
                best_path=path
        best_l_list.append(best_l)

        for edge in edges_in_path(best_path):
            cur_F[edge] = cur_F[edge[::-1]] = (1-p)*cur_F[edge] +p/best_l


        count+=1
        #print(c, best_l,best_l_list[-2])
        if best_l == best_l_list[-2]:
            c+=1
            if c==100:
                print('Выход по времени')
                break
        else:
            c=0
        print(count,': ',best_l)
    #print(l_list)
    #print(cur_F)
    return (best_l, best_path,best_l_list)

def ant_colony_system_drawing(points,method_coef,ph_coef,b=1,p=0.5,q=1):
    # method_coef - коэффициент выбора метода выбора вершины
    # ph_coef - коэффициент локального обновления феромона
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона

    M = generate_weights(points)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.ion()
    minx=min([i[0]for i in points])
    miny=min([i[1]for i in points])
    maxx=max([i[0]for i in points])
    maxy=max([i[1]for i in points])
    plt.axes([minx,miny,maxx,maxy])

    count=0
    init_pos = set(range(len(M)))
    best_l=float('inf')
    best_path=[]
    #l_list=[]
    best_l_list=[best_l]
    cur_F=np.zeros(np.shape(M), dtype = np.float)
    cur_F.fill(3/(nearest_neighbour_tour(M,0)[0]))
    c=0
    while count<=150: # внешний цикл по кол-ву итераций
        for ant in range(len(M)): # цикл по муравьям(в каждом городе по муравью)
            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l, new_F = ant_colony_system_path(cur,1,b,q,M,cur_F,pos,method_coef,ph_coef) # строим путь, считаем длину и получаем новую матрицу феромона

            #l_list.append(l)

            if l<=best_l:
                best_l=l
                best_path=path
            best_l_list.append(best_l)

            b_pts = points_in_path(best_path).transpose()
            pts = points_in_path(path).transpose()
            ax1.cla()
            ax1.plot(points.transpose()[0],points.transpose()[1],'ro',b_pts[0],b_pts[1],'b',linewidth = 3.0)
            ax1.plot(pts[0],pts[1],'g')
            ax1.text(2, 5, str(count)+ '  '+ str(best_l))
            fig.canvas.draw()
            plt.pause(0.01)
            plt.show()

        for edge in edges_in_path(best_path):
            new_F[edge] = new_F[edge[::-1]] = (1-p)*new_F[edge] +p/best_l

        cur_F=new_F
        count+=1
        if best_l==best_l_list[-2]:
            c+=1
            if c==10:
                print('Выход по времени')
                break
        else:
            c=0
        #print(count,': ',best_l)
    #print(l_list)
    #print(cur_F)
    return (best_l, best_path,best_l_list)

def mm_ant_system_path(cur,a,b,M,F,pos): # построить путь для одного муравья
    l=0 # начальная длина пути
    path=[cur] # пройденный путь
    while pos!=set():
        new_cur=choose_vertex(cur,a,b,M,F,pos) # выбираем новую текущую вершину
        l+=M[cur,new_cur] # увеличиваем длину пути
        cur=new_cur
        pos.remove(cur)
        path.append(cur) # добавляем текущую вершину в путь
    path.append(path[0]) # замыкаем путь муравья
    l+=M[cur,path[0]]
    return (path, l)

def mm_ant_system(M,a=1,b=1,p=0.02):
    # a - степень при феромоне
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона
    # e - количество элитных муравьев

    count=0 # счетчик итераций
    n=len(M) # количество городов в графе
    neighbour_tour = nearest_neighbour_tour(M,0) # тур по ближайсши соседним города
    init_pos = set(range(len(M))) # изначально доступные для посещения города
    best_l=neighbour_tour[0]
    best_path=neighbour_tour[1]
    l_list=[]
    best_l_list=[best_l]
    cur_best_l_list = []
    init_max = 1/(p*best_l) # инициализация максимального количества феромона
    t_max = init_max
    init_min = t_max*(1-0.05**(1/n))/((n/2-1)*0.05**(1/n)) # инициализация минимального количества феромона
    t_min = init_min

    cur_F=np.zeros(np.shape(M), dtype = np.float)
    cur_F.fill(t_max) # инициализация феромона на ребрах графа
    cur_F =cur_F - np.diag(np.diag(cur_F))

    stag_count = 0 # счетчик застоя
    while count<=250: # внешний цикл по кол-ву итераций
        for ant in range(n): # цикл по муравьям(в каждом городе по муравью)
            t_max = 1/(p*best_l) # реинициализация макс. кол-ва феромона на арке
            t_min = t_max*(1-0.05**(1/n))/((n/2-1)*0.05**(1/n)) # реинициализация мин. кол-ва феромона на арке

            cur_best_l = float('inf') # задание лучшей длиныпути на данной итерации
            cur_best_path = [] # задание лучшего пути на данной итерации

            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l = mm_ant_system_path(cur,a,b,M,cur_F,pos) # строим путь, считаем длину

            l_list.append(l)

            if l<=cur_best_l: # проверка на уменьшение лучшего пути на итерации
                cur_best_l=l
                cur_best_path=path
            if l<=best_l: # проверка на уменьшение глобального лучшего пути
                best_l=l
                best_path=path

        best_l_list.append(best_l)
        cur_best_l_list.append(cur_best_l)
        cur_F=(1-p)*cur_F # происходит испарение феромона по всем ребрам

        for edge in edges_in_path(cur_best_path): # добавление феромона на лучший путь в итерации
            cur_F[edge] =cur_F[edge[::-1]] = cur_F[edge] +1/cur_best_l

        count+=1

        if best_l==best_l_list[-2]: # если происходит застой
            stag_count+=1
            if stag_count==50:
                cur_F.fill(t_max)
                cur_F =cur_F - np.diag(np.diag(cur_F))
                stag_count=0
        else:
            stag_count=0
        print(count,': ',best_l)
    #print(l_list)
    print(cur_F)
    print((t_max,t_min))
    return (best_l, best_path,best_l_list,cur_best_l_list)

def mm_ant_system_drawing(points,a=1,b=1,p=0.02):
    # a - степень при феромоне
    # b - степень при весе ребра
    # p - коэффициент испарения
    # q - коэффициент для приращения феромона
    # e - количество элитных муравьев
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.ion()
    count=0 # счетчик итераций
    M = generate_weights(points)
    n=len(M) # количество городов в графе
    neighbour_tour = nearest_neighbour_tour(M,0) # тур по ближайсши соседним города
    init_pos = set(range(len(M))) # изначально доступные для посещения города
    best_l=neighbour_tour[0]
    best_path=neighbour_tour[1]
    l_list=[]
    best_l_list=[best_l]
    cur_best_l_list = []
    init_max = 1/(p*best_l) # инициализация максимального количества феромона
    t_max = init_max
    init_min = t_max*(1-0.05**(1/n))/((n/2-1)*0.05**(1/n)) # инициализация минимального количества феромона
    t_min = init_min

    cur_F=np.zeros(np.shape(M), dtype = np.float)
    cur_F.fill(t_max) # инициализация феромона на ребрах графа
    cur_F =cur_F - np.diag(np.diag(cur_F))

    stag_count = 0 # счетчик застоя
    while count<=250: # внешний цикл по кол-ву итераций
        for ant in range(n): # цикл по муравьям(в каждом городе по муравью)
            t_max = 1/(p*best_l) # реинициализация макс. кол-ва феромона на арке
            t_min = t_max*(1-0.05**(1/n))/((n/2-1)*0.05**(1/n)) # реинициализация мин. кол-ва феромона на арке

            cur_best_l = float('inf') # задание лучшей длиныпути на данной итерации
            cur_best_path = [] # задание лучшего пути на данной итерации

            cur=ant # инициализация текущей вершины
            pos = init_pos.copy()
            pos.remove(cur)

            path, l = mm_ant_system_path(cur,a,b,M,cur_F,pos) # строим путь, считаем длину

            l_list.append(l)

            if l<=cur_best_l: # проверка на уменьшение лучшего пути на итерации
                cur_best_l=l
                cur_best_path=path
            if l<=best_l: # проверка на уменьшение глобального лучшего пути
                best_l=l
                best_path=path
        ax1.cla()
        b_pts = points_in_path(points,best_path).transpose()
        pts = points_in_path(points,path).transpose()
        ax1.plot(points.transpose()[0],points.transpose()[1],'ro',b_pts[0],b_pts[1],'b',linewidth = 3.0)
        #ax1.plot(pts[0],pts[1],'g')
        ax1.text(2, 5, str(count)+ '  '+ str(best_l))
        fig.canvas.draw()
        plt.pause(0.05)
        plt.show()

        best_l_list.append(best_l)
        cur_best_l_list.append(cur_best_l)
        cur_F=(1-p)*cur_F # происходит испарение феромона по всем ребрам

        for edge in edges_in_path(cur_best_path): # добавление феромона на лучший путь в итерации
            cur_F[edge] =cur_F[edge[::-1]] = cur_F[edge] +1/cur_best_l

        count+=1

        if best_l==best_l_list[-2]: # если происходит застой
            stag_count+=1
            if stag_count==50:
                cur_F.fill(t_max)
                cur_F =cur_F - np.diag(np.diag(cur_F))
                stag_count=0
        else:
            stag_count=0
        print(count,': ',best_l)
    #print(l_list)
    print(cur_F)
    print((t_max,t_min))
    return (best_l, best_path,best_l_list,cur_best_l_list)

if __name__=='__main__':
    #rand.seed(13)
    #rand.seed(1289)
    rand.seed(123)
    points = generate_points(1,100,20)
    M6 = generate_weights(points)

    textfile = open('mat2.txt', 'w')
    for i in M6:
        for el in i:
            textfile.write(str(el)+' ')
        textfile.write('\n')
    textfile.close()

    with open('pts2.txt','w') as f:
        for i in points:
            f.write(str(i)[1:-2]+'\n')
    rand.seed(126368)
    st = time.time()
    #ants = ant_colony_system(M6,b=4,method_coef=0.9,ph_coef=0.5,q=1/nearest_neighbour_tour(M6,0)[0])
    #ants = mm_ant_system_drawing(points,b=4)
    ants = mm_ant_system(M6,b=4)
    #ants = ant_system(M6,e=5,q=10)
    en=time.time()
    path = ants[1]
    print(en-st)
    print(ants)

    b_pts = points_in_path(points,path).transpose()
    #print(b_pts)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(points.transpose()[0],points.transpose()[1],'ro',b_pts[0],b_pts[1],'b',linewidth = 3.0)
    while True:
        plt.pause(5)
