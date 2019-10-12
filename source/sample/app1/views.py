from django.shortcuts import render
from django.views import generic
from .forms import InputForm
from app1.utils import *
from app1.config import *
from IPython.display import HTML


class InputView(generic.FormView):
    form_class = InputForm
    template_name = 'app1/demo.html'

    def form_valid(self, form):
        input_seq = self.request.POST["messages"] #画面からの入力文章を取得
        TEXT = pickle_load(pkl_path)   #vocabデータのロード
        net_trained = load_model(model_path, TEXT)  #学習済みモデルのロード
        input, input_mask = create_input_data(input_seq, TEXT)  #モデルインプット用のデータを生成
        outputs, normlized_weights_1, normlized_weights_2 = net_trained(input, input_mask)  #予測
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        html_output = mk_html(input, preds, normlized_weights_1, normlized_weights_2, TEXT)  # HTML作成
  

        context = {
            'input_seq': input_seq,
            'html_output': html_output,
        }
        return render(self.request, 'app1/demo.html', context)

    def form_invalid(self, form):
        return render(self.request, 'app1/demo.html', {'form': form})

